"""layout-v2：共用模板、使用次数、覆盖/成本对照与短维度公式回归。"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[2]
SRC = WORKSPACE / "structure_reader" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openpyxl import load_workbook  # noqa: E402

from structure_reader.export.deepseek_perf_layout_v2 import (  # noqa: E402
    MAIN_HEADERS,
    _build_plan,
    build_perf_layout_v2_workbook,
)
from structure_reader.export.hf_workbook import (  # noqa: E402
    build_structures_workbook_spec,
    export_structures_workbook,
)

FROZEN = Path(os.environ.get("E01_FROZEN_DIR", WORKSPACE / "inputs" / "frozen"))


def _load():
    if not (FROZEN / "layer_structures.json").is_file():
        return None, None
    return (json.loads((FROZEN / "layer_structures.json").read_text(encoding="utf-8")),
            json.loads((FROZEN / "network_overview.json").read_text(encoding="utf-8")))


def _header(ws):
    for row in range(1, ws.max_row + 1):
        if ws.cell(row, 1).value == "结构/通路":
            return row
    raise AssertionError("no main header")


def _op_rows(ws, op_id):
    return [row for row in range(1, ws.max_row + 1)
            if ws.cell(row, 2).value == op_id]


@unittest.skipUnless(FROZEN.is_dir(), "缺少 E01 冻结输入；如实跳过")
class LayoutV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.container, cls.network = _load()
        cls.spec = build_structures_workbook_spec(cls.container, cls.network)
        cls.plan = _build_plan(cls.spec)
        cls.payload = build_perf_layout_v2_workbook(cls.container, cls.network)
        cls.wb = load_workbook(io.BytesIO(cls.payload), data_only=False)

    def test_sheets_and_12_columns(self):
        self.assertEqual(self.wb.sheetnames[:5],
                         ["Prefill", "Decode", "模板映射", "参数与来源", "项分解"])
        self.assertEqual(self.wb.sheetnames[5], "原始参数")
        for name in ("Prefill", "Decode"):
            ws = self.wb[name]
            header = _header(ws)
            self.assertEqual(tuple(ws.cell(header, c).value for c in range(1, 13)),
                             MAIN_HEADERS)
            self.assertEqual(ws.max_column, 12)
            self.assertLess(ws.max_row, 220)
            for row in range(1, ws.max_row + 1):
                for col in range(13, ws.max_column + 1):
                    self.assertIsNone(ws.cell(row, col).value)

    def test_block_headers_only_in_A_once(self):
        ws = self.wb["Prefill"]
        blocks = [block for block, _usage, _items in self.plan.blocks]
        for block in blocks:
            self.assertEqual(sum(1 for row in range(1, ws.max_row + 1)
                                 if str(ws.cell(row, 1).value).startswith(block)),
                             1, block)
        # 模板行 A 为空，没有逐行结构归属标签。
        for row in _op_rows(ws, "q_a_proj"):
            self.assertIsNone(ws.cell(row, 1).value)
        self.assertFalse(any(str(ws.cell(row, 1).value).startswith("层组 ")
                             for row in range(1, ws.max_row + 1)))

    def test_shared_templates_once_and_usage(self):
        ws = self.wb["Prefill"]
        self.assertEqual(len(_op_rows(ws, "attn_hc_project")), 1)
        self.assertEqual(len(_op_rows(ws, "q_a_proj")), 1)
        self.assertEqual(len(_op_rows(ws, "cache_update")), 3)  # SWA/CSA/HCA
        labels = [str(ws.cell(row, 1).value) for row in range(1, ws.max_row + 1)]
        self.assertTrue(any(label.startswith("共用 mHC（×86 次）") for label in labels))
        self.assertTrue(any(label.startswith("Attention Q/KV 前置（×43 次）") for label in labels))
        self.assertTrue(any(label.startswith("共用残差混合（×86 次）") for label in labels))

    def test_coverage_and_cost_comparison(self):
        self.assertEqual(self.plan.coverage["original_instances"], 291)
        self.assertEqual(self.plan.coverage["billable_instances"], 282)
        self.assertEqual(self.plan.coverage["excluded_instances"], 8)
        self.assertEqual(self.plan.coverage["container_instances"], 1)
        self.assertEqual(self.plan.coverage["templates"], 101)
        mapping = self.wb["模板映射"]
        header = next(row for row in range(1, mapping.max_row + 1)
                      if mapping.cell(row, 1).value == "原 op")
        kinds = [mapping.cell(row, 4).value for row in range(header + 1, mapping.max_row + 1)
                 if isinstance(mapping.cell(row, 1).value, str)
                 and mapping.cell(row, 4).value in ("共用/差异模板", "excluded", "container")]
        self.assertEqual(kinds.count("共用/差异模板"), 282)
        self.assertEqual(kinds.count("excluded"), 8)
        self.assertEqual(kinds.count("container"), 1)
        # 成本对照表差额全 0（构建期 N_hit=6）。
        diffs = []
        for row in range(1, mapping.max_row + 1):
            if isinstance(mapping.cell(row, 3).value, (int, float)) and \
                    isinstance(mapping.cell(row, 4).value, (int, float)) and \
                    mapping.cell(row, 2).value in ("cube", "vec", "op"):
                diffs.append(mapping.cell(row, 5).value)
        self.assertTrue(diffs)
        self.assertTrue(all(value == 0 for value in diffs), diffs)

    def test_dims_are_short_direct_and_n_is_head_or_one(self):
        ws = self.wb["Prefill"]
        for row in range(_header(ws) + 1, ws.max_row + 1):
            for col in range(4, 10):
                value = ws.cell(row, col).value
                if isinstance(value, str) and value.startswith("="):
                    self.assertNotIn("IF(", value)
                    self.assertNotIn("ISNUMBER", value)
                    self.assertLessEqual(len(value), 40, (row, col, value))
        qa = _op_rows(ws, "q_a_proj")[0]
        self.assertEqual(ws.cell(qa, 4).value, 1)
        qk = _op_rows(ws, "qk_scaled")[0]
        self.assertIn("$", str(ws.cell(qk, 4).value))
        for row in range(_header(ws) + 1, ws.max_row + 1):
            n = ws.cell(row, 4).value
            if isinstance(n, str) and n.startswith("="):
                self.assertIn("$", n)

    def test_units_batch_and_throughput_guards(self):
        for name in ("Prefill", "Decode"):
            ws = self.wb[name]
            for cell in ("K6", "K7"):
                self.assertIn("element/ms", str(ws[cell].value))
            self.assertEqual(ws["C8"].value, "固定 1")
            import re
            refs = [ws.cell(row, col).coordinate
                    for row in range(1, ws.max_row + 1)
                    for col in range(1, 13)
                    if isinstance(ws.cell(row, col).value, str)
                    and re.search(r"(?<!!)\$C\$8", ws.cell(row, col).value)]
            self.assertEqual(refs, [])
            qa = _op_rows(ws, "q_a_proj")[0]
            qn = _op_rows(ws, "q_a_norm")[0]
            self.assertIn("$J$6>0", str(ws.cell(qa, 10).value))
            self.assertIn("$J$7>0", str(ws.cell(qn, 11).value))

    def test_composite_repeat_and_derived_params(self):
        ws = self.wb["Prefill"]
        scorer = _op_rows(ws, "i_scorer")[0]
        self.assertIn("项分解", str(ws.cell(scorer, 10).value))
        shared = _op_rows(ws, "shared_gate_up")[0]
        self.assertIn(")*2", str(ws.cell(shared, 10).value))
        params = []
        for row in range(1, _header(ws)):
            params.extend((ws.cell(row, 3).value, ws.cell(row, 10).value))
        for name in ("M_attn", "M_og", "M_route", "M_res", "K_og", "K_hc",
                     "N_hc", "N_qkv", "N_slice", "Kv_vis", "H_bias_on"):
            self.assertIn(name, params)

    def test_validated_inputs_are_in_parameter_layer(self):
        ws = self.wb["Prefill"]
        params = set()
        for row in range(1, _header(ws)):
            params.update((ws.cell(row, 3).value, ws.cell(row, 10).value))
        for name in ("V_seqlen", "V_kvcach", "V_Nhit", "V_H", "V_G", "V_Nh", "V_M"):
            self.assertIn(name, params)
        qa = _op_rows(ws, "q_a_proj")[0]
        self.assertIn("$K$", str(ws.cell(qa, 5).value))  # V_seqlen 参数层
        self.assertIn("$K$", str(ws.cell(qa, 6).value))  # V_H
        # 基础参数格本身是校验公式，原始值在隐藏的原始参数表。
        h_row = next(row for row in range(1, _header(ws))
                     if ws.cell(row, 3).value == "H")
        self.assertIn("原始参数", str(ws.cell(h_row, 4).value))

    def test_deterministic_and_legacy_stable(self):
        again = build_perf_layout_v2_workbook(self.container, self.network)
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(),
                         hashlib.sha256(again).hexdigest())
        first = export_structures_workbook(self.container, self.network)
        second = export_structures_workbook(self.container, self.network)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
