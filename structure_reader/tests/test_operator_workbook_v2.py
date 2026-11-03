"""T04 通用工作簿导出回归：声明式 sheet/分组、可填写配置、公式与未知。

覆盖：单页 fixture（证明导出器不再固定 Prefill/Decode）、真实 DeepSeek 两页四组 + 整网外围
（可填写 seqlen/kvcachelen/B/精度/位宽/CUBE 与 VEC 吞吐、派生值公式、跨组隔离、循环归属、
CUBE/VEC 分类与维度、用户规则耗时公式、层组小计与串行汇总、未知不默认 0）、
合成小例（单层小计×层数只乘一次、repeat 符号守卫）、稳定字节与具名拒绝。

运行：
  cd structure_reader && PYTHONPATH=src python3 tests/test_operator_workbook_v2.py
"""

from __future__ import annotations

import copy
import io
import json
import re
import sys
import zipfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import unittest  # noqa: E402

from openpyxl import load_workbook  # noqa: E402

from structure_reader.export.operator_workbook import (  # noqa: E402
    WorkbookExportError,
    export_workbook,
)
from structure_reader.scenario_evaluator import (  # noqa: E402
    excel_formula,
    expression_names,
)
from structure_reader.extract.hf_scenario import evaluate_derivations  # noqa: E402

MODEL_KEY = "deepseek_v4_flash_0731_hf"


def _workspace_ready() -> bool:
    return (REPO_ROOT / "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json").is_file() and (
        REPO_ROOT / "hflib/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py"
    ).is_file()


def _minimal_spec() -> dict:
    """明确的一页 fixture：单 sheet、单分组、两个派生公式；不代表 HiDream 已接入。"""
    return {
        "schema": "operator_workbook.v2",
        "model": {
            "model_key": "fixture_single_page",
            "display_name": "单页 fixture 模型",
            "implementation": "transformers",
            "source_revision": "fixture",
            "config_path": "fixtures/config.json",
        },
        "performance": {"status": "待提供", "note": "fixture 不含性能公式"},
        "sheets": [{
            "id": "single",
            "name": "Layer1",
            "title": "单页能力 fixture",
            "notes": ["只验证声明的单页/单组导出能力。"],
            "inputs": [
                {"id": "seqlen", "symbol": "seqlen", "label": "seqlen", "value": 8,
                 "kind": "integer", "source": "fixture", "note": "可改", "editable": True},
                {"id": "kvcachelen", "symbol": "kvcachelen", "label": "kvcachelen",
                 "value": 0, "kind": "integer", "source": "fixture", "note": "可改",
                 "editable": True},
                {"id": "batch_size", "symbol": "B", "label": "batch", "value": 1,
                 "kind": "integer", "source": "fixture", "note": "可改", "editable": True},
                {"id": "precision", "symbol": "精度", "label": "精度", "value": "FP32",
                 "kind": "text", "source": "fixture", "note": "可改", "editable": True},
            ],
            "groups": [{
                "id": "g1",
                "label": "Fixture 组",
                # 通用单页 fixture 故意不声明 Decoder 层号/代表层：
                # 不伪造 HF 字段，只有声明分组与循环。
                "loop": {"count": 1, "label": "单页", "condition": "一次", "note": ""},
                "template_note": "fixture 模板不共享权重",
                "scope_note": "fixture 单组",
                "parameters": [
                    {"id": "B", "symbol": "B", "meaning": "batch", "value": None,
                     "expression": "引用本表 batch_size", "note": "", "source_refs": [],
                     "value_ref": "batch_size", "formula": None, "unknown_reason": ""},
                    {"id": "seqlen", "symbol": "seqlen", "meaning": "seqlen", "value": None,
                     "expression": "引用本表 seqlen", "note": "", "source_refs": [],
                     "value_ref": "seqlen", "formula": None, "unknown_reason": ""},
                    {"id": "kvcachelen", "symbol": "kvcachelen", "meaning": "history",
                     "value": None, "expression": "引用本表 kvcachelen", "note": "",
                     "source_refs": [], "value_ref": "kvcachelen", "formula": None,
                     "unknown_reason": ""},
                    {"id": "H", "symbol": "H", "meaning": "hidden", "value": 8,
                     "expression": None, "note": "", "source_refs": [],
                     "value_ref": None, "formula": None, "unknown_reason": ""},
                    {"id": "H_other", "symbol": "H", "meaning": "同名但不被引用的 H",
                     "value": 16, "expression": None, "note": "", "source_refs": [],
                     "value_ref": None, "formula": None, "unknown_reason": ""},
                    {"id": "Nwin", "symbol": "N_win", "meaning": "窗口数", "value": None,
                     "expression": "seqlen // 4", "note": "", "source_refs": [],
                     "value_ref": None, "formula": "seqlen // 4", "unknown_reason": ""},
                    {"id": "Lkv", "symbol": "L_kv", "meaning": "KV 长度", "value": None,
                     "expression": "min(seqlen, 8) + kvcachelen", "note": "",
                     "source_refs": [], "value_ref": None,
                     "formula": "min(seqlen, 8) + kvcachelen", "unknown_reason": ""},
                ],
                "tensors": [
                    {"id": "t_x", "name": "x", "shape": ["B", "seqlen", "H"],
                     "parameter_refs": ["B", "seqlen", "H"], "kind": "activation", "note": ""},
                    {"id": "t_y", "name": "y", "shape": ["B", "seqlen", "H"],
                     "parameter_refs": ["seqlen", "H"], "kind": "activation", "note": ""},
                ],
                "operators": [{
                    "id": "op1", "module_path": "Fixture.forward", "name": "fixture op",
                    "op": "linear", "inputs": ["t_x"], "outputs": ["t_y"], "weights": [],
                    "parameter_refs": ["H"], "source_refs": [],
                    "analysis_status": "analyzed", "note": "",
                }],
                "cache_notes": [],
                "coverage_notes": [],
            }],
            "network": None,
        }],
    }


def _param_header_row(ws, occurrence: int = 0) -> int:
    rows = [r for r in range(1, ws.max_row + 1) if ws.cell(r, 1).value == "参数id"]
    return rows[occurrence]


def _param_rows(ws, header: int) -> list[int]:
    rows = []
    for r in range(header + 1, ws.max_row + 1):
        value = ws.cell(r, 1).value
        if not isinstance(value, str) or not value:
            break
        rows.append(r)
    return rows


def _input_rows(ws) -> dict[str, int]:
    header = next(r for r in range(1, ws.max_row + 1) if ws.cell(r, 1).value == "配置id")
    rows = {}
    for r in range(header + 1, ws.max_row + 1):
        key = ws.cell(r, 1).value
        if not isinstance(key, str) or not key:
            break
        rows[key] = r
    return rows


def _formula_targets(formula: str) -> list[tuple[str, int]]:
    return [(col, int(row)) for col, row in re.findall(r"\$([A-Z]+)\$(\d+)", formula or "")]


def _iferror_fallbacks(text: str) -> list[str]:
    """IFERROR(expr, fallback) 的 fallback 文本（用于断言未知没有被折成 0）。"""
    fallbacks = []
    index = 0
    while True:
        start = text.find("IFERROR(", index)
        if start < 0:
            return fallbacks
        depth = 0
        commas = []
        for position in range(start + len("IFERROR"), len(text)):
            char = text[position]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    args = [(commas[i] + 1, commas[i + 1])
                            for i in range(len(commas) - 1)]
                    if commas:
                        last = commas[-1]
                        fallbacks.append(text[last + 1:position].strip())
                    index = position
                    break
            elif char == "," and depth == 1:
                commas.append(position)
        else:
            return fallbacks


class TestSinglePageFixture(unittest.TestCase):
    def test_generic_export_supports_declared_single_sheet(self):
        payload = export_workbook(_minimal_spec())
        wb = load_workbook(io.BytesIO(payload))
        self.assertEqual(wb.sheetnames, ["Layer1"])
        ws = wb["Layer1"]
        inputs = _input_rows(ws)
        self.assertEqual(set(inputs), {"seqlen", "kvcachelen", "batch_size", "precision"})
        header = _param_header_row(ws)
        rows = _param_rows(ws, header)
        by_id = {ws.cell(r, 1).value: r for r in rows}
        # 派生的两个值是真公式，引用本表输入/本组参数单元格。
        self.assertTrue(str(ws.cell(by_id["Nwin"], 4).value).startswith("="))
        targets = set(_formula_targets(str(ws.cell(by_id["Nwin"], 4).value)))
        self.assertIn(("D", inputs["seqlen"]), targets)
        lkv = str(ws.cell(by_id["Lkv"], 4).value)
        self.assertTrue(lkv.startswith("="))
        lkv_targets = set(_formula_targets(lkv))
        self.assertIn(("D", inputs["seqlen"]), lkv_targets)
        self.assertIn(("D", inputs["kvcachelen"]), lkv_targets)
        self.assertEqual(ws.cell(by_id["Nwin"], 5).value, "派生（Excel 公式）")

    def test_actual_shape_uses_explicit_refs_for_duplicate_symbol(self):
        """同名 H 有两个候选，但 op/tensor 显式只引用 H：实际 shape 必须绑定 H=8。"""
        payload = export_workbook(_minimal_spec())
        ws = load_workbook(io.BytesIO(payload))["Layer1"]
        header = _param_header_row(ws)
        rows = _param_rows(ws, header)
        by_id = {ws.cell(r, 1).value: r for r in rows}
        op_hdr = next(r for r in range(1, ws.max_row + 1)
                      if ws.cell(r, 1).value == "黑盒标记")
        actual = str(ws.cell(op_hdr + 1, 17).value)
        self.assertTrue(actual.startswith("="), actual)
        targets = set(_formula_targets(actual))
        self.assertIn(("D", by_id["H"]), targets)
        self.assertNotIn(("D", by_id["H_other"]), targets)
        self.assertNotIn("歧义", actual)

    def test_actual_shape_duplicate_without_explicit_ref_is_unknown(self):
        """没有显式引用可分辨同名 H 时，实际 shape 写歧义原因，不猜值。"""
        spec = _minimal_spec()
        group = spec["sheets"][0]["groups"][0]
        group["operators"][0]["parameter_refs"] = []
        for tensor in group["tensors"]:
            tensor["parameter_refs"] = []
        ws = load_workbook(io.BytesIO(export_workbook(spec)))["Layer1"]
        op_hdr = next(r for r in range(1, ws.max_row + 1)
                      if ws.cell(r, 1).value == "黑盒标记")
        actual = str(ws.cell(op_hdr + 1, 17).value)
        self.assertIn("歧义", actual)
        self.assertNotIn("H_other", actual)

    def test_actual_shape_duplicate_with_multiple_explicit_refs_is_unknown(self):
        """显式引用同时命中两个同名候选时仍为歧义，不能任选其一。"""
        spec = _minimal_spec()
        group = spec["sheets"][0]["groups"][0]
        group["operators"][0]["parameter_refs"] = ["H", "H_other"]
        ws = load_workbook(io.BytesIO(export_workbook(spec)))["Layer1"]
        op_hdr = next(r for r in range(1, ws.max_row + 1)
                      if ws.cell(r, 1).value == "黑盒标记")
        actual = str(ws.cell(op_hdr + 1, 17).value)
        self.assertIn("歧义", actual)
        self.assertNotIn("H_other", actual)

    def test_single_page_without_layer_fields_has_no_fake_decoder_scope(self):
        """通用单页 fixture 不声明 layer_indices/代表层，仍可导出且不伪造层号。"""
        spec = _minimal_spec()
        group = spec["sheets"][0]["groups"][0]
        self.assertNotIn("layer_indices", group)
        self.assertNotIn("representative_layer_index", group)
        ws = load_workbook(io.BytesIO(export_workbook(spec)))["Layer1"]
        texts = " ".join(str(c.value) for row in ws.iter_rows() for c in row if c.value)
        self.assertIn("未声明 Decoder 层号", texts)
        self.assertNotIn("真实层 [0]", texts)
        # 实际 shape 列也是活公式，不是只拼符号。
        header = _param_header_row(ws)
        rows = _param_rows(ws, header)
        by_id = {ws.cell(r, 1).value: r for r in rows}
        op_hdr = next(r for r in range(1, ws.max_row + 1)
                      if ws.cell(r, 1).value == "黑盒标记")
        op_row = op_hdr + 1
        for col in (17, 18):
            cell = ws.cell(op_row, col)
            self.assertEqual(cell.data_type, "f", (col, cell.value))
            self.assertIn("IF(", str(cell.value))
            self.assertIn('"["', str(cell.value))
        self.assertEqual(str(ws.cell(op_row, 19).value), '="（无）"')

    def test_generic_network_summary_uses_actual_group_count(self):
        spec = _minimal_spec()
        spec["sheets"][0]["network"] = {
            "label": "单组网络归属",
            "periphery": [],
            "groups": [{"id": "g1", "label": "Fixture 组", "count": 1}],
            "layer_count": 1,
            "notes": ["单组 fixture 不写死四组。"],
            "cost_status": "待提供",
        }
        ws = load_workbook(io.BytesIO(export_workbook(spec)))["Layer1"]
        texts = " ".join(str(c.value) for row in ws.iter_rows() for c in row if c.value)
        self.assertIn("1 组层数合计 1", texts)
        self.assertNotIn("四组层数合计", texts)

    def test_declared_sheet_count_is_not_hardcoded(self):
        spec = _minimal_spec()
        second = copy.deepcopy(spec["sheets"][0])
        second["id"] = "second"
        second["name"] = "Layer2"
        second["title"] = "第二页"
        spec["sheets"].append(second)
        wb = load_workbook(io.BytesIO(export_workbook(spec)))
        self.assertEqual(wb.sheetnames, ["Layer1", "Layer2"])


class TestFormulaCompiler(unittest.TestCase):
    def test_whitelist_semantics_match_evaluator(self):
        refs = {"a": "$D$1", "b": "$D$2"}
        self.assertEqual(excel_formula("a // b", refs), "INT(($D$1)/($D$2))")
        self.assertEqual(excel_formula("a % b", refs), "MOD($D$1,$D$2)")
        self.assertEqual(excel_formula("min(a, b)", refs), "MIN($D$1,$D$2)")
        self.assertEqual(excel_formula("a != b and a > 0", refs), "AND(($D$1<>$D$2),($D$1>0))")
        self.assertEqual(expression_names("min(a, b) + 2"), ["a", "b"])
        with self.assertRaises(ValueError):
            excel_formula("sqrt(a)", refs)
        with self.assertRaises(ValueError):
            excel_formula("a + missing", refs)


class TestRejectedSpecs(unittest.TestCase):
    def expect_reject(self, editor, needle: str):
        spec = _minimal_spec()
        editor(spec)
        with self.assertRaises(WorkbookExportError) as caught:
            export_workbook(spec)
        self.assertIn(needle, str(caught.exception))

    def test_duplicate_sheet_and_group_ids(self):
        self.expect_reject(lambda s: s["sheets"].append(copy.deepcopy(s["sheets"][0])), "工作表")
        self.expect_reject(
            lambda s: s["sheets"][0]["groups"].append(copy.deepcopy(s["sheets"][0]["groups"][0])),
            "分组 id")

    def test_duplicate_parameter_id_and_unresolved_refs(self):
        def dup_param(spec):
            group = spec["sheets"][0]["groups"][0]
            group["parameters"].append(copy.deepcopy(group["parameters"][0]))

        self.expect_reject(dup_param, "参数 id")

        def bad_value_ref(spec):
            spec["sheets"][0]["groups"][0]["parameters"][0]["value_ref"] = "no_such_input"

        self.expect_reject(bad_value_ref, "不存在")

        def bad_tensor_ref(spec):
            spec["sheets"][0]["groups"][0]["operators"][0]["inputs"] = ["t_missing"]

        self.expect_reject(bad_tensor_ref, "输入张量")

    def test_unresolved_formula_is_unknown_not_zero(self):
        spec = _minimal_spec()
        param = spec["sheets"][0]["groups"][0]["parameters"][-1]
        param["formula"] = "seqlen + missing_var"
        ws = load_workbook(io.BytesIO(export_workbook(spec)))["Layer1"]
        header = _param_header_row(ws)
        rows = _param_rows(ws, header)
        by_id = {ws.cell(r, 1).value: r for r in rows}
        cell = ws.cell(by_id["Lkv"], 4)
        self.assertEqual(cell.value, "未知（缺少参数: missing_var）")
        self.assertEqual(cell.data_type, "s")
        self.assertNotIn("=0", str(cell.value))


class TestStableBytes(unittest.TestCase):
    def test_same_spec_same_bytes_and_key_order_independent(self):
        first = export_workbook(_minimal_spec())
        self.assertEqual(export_workbook(_minimal_spec()), first)

        def reversed_keys(value):
            if isinstance(value, dict):
                return {k: reversed_keys(v) for k, v in reversed(list(value.items()))}
            if isinstance(value, list):
                return [reversed_keys(v) for v in value]
            return value

        self.assertEqual(export_workbook(reversed_keys(_minimal_spec())), first)
        with zipfile.ZipFile(io.BytesIO(first)) as zf:
            self.assertIsNone(zf.testzip())
            self.assertEqual({i.date_time for i in zf.infolist()},
                             {(1980, 1, 1, 0, 0, 0)})


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入；如实跳过")
class TestDeepSeekWorkbook(unittest.TestCase):
    """真实结构的四组两页工作簿：可填写配置、公式引用、循环归属、成本待提供。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_layer_documents
        from structure_reader.extract.deepseek_v4_network import build_network_overview
        from structure_reader.extract.hf_layer_structures import build_structures_container
        from structure_reader.export.hf_workbook import (
            build_structures_workbook_spec,
            build_structure_workbook_spec,
        )

        cls.container = build_structures_container(REPO_ROOT)
        cls.network = build_network_overview(REPO_ROOT)
        cls.spec = build_structures_workbook_spec(cls.container, cls.network)
        cls.payload = export_workbook(cls.spec)
        cls.wb = load_workbook(io.BytesIO(cls.payload))
        cls.structure, _document = build_layer_documents(REPO_ROOT, layer_index=4)
        cls.single_spec = build_structure_workbook_spec(cls.structure)
        cls.single_wb = load_workbook(io.BytesIO(export_workbook(cls.single_spec)))

    def test_two_text_sheets_layer_groups_plus_periphery(self):
        self.assertEqual(self.wb.sheetnames, ["Prefill", "Decode"])
        for name in self.wb.sheetnames:
            ws = self.wb[name]
            headers = [r for r in range(1, ws.max_row + 1) if ws.cell(r, 1).value == "参数id"]
            # 四类真实层组 + 整网外围块（入口/出口外围算子同源 network_overview）。
            self.assertEqual(len(headers), 5)
            labels = [ws.cell(h - 11, 1).value for h in headers]
            self.assertTrue(all(str(label).startswith("层组 ") for label in labels), labels)
            self.assertIn("整网外围", labels[-1])
            # 每个块都写 CUBE/VEC 性能列，且只有 5 个参数区/算子区。
            self.assertEqual(sum(1 for r in range(1, ws.max_row + 1)
                                 if ws.cell(r, 1).value == "黑盒标记"), 5)

    def test_deepseek_adapter_still_requires_real_layer_attribution(self):
        from structure_reader.export.hf_workbook import build_structures_workbook_spec

        container = copy.deepcopy(self.container)
        del container["variants"][0]["layer_indices"]
        with self.assertRaises(ValueError) as caught:
            build_structures_workbook_spec(container)
        self.assertIn("layer_indices", str(caught.exception))

    def test_actual_shape_columns_are_recaclable_or_explicit_unknown(self):
        """实际 shape 列：可推导的为活公式，数据相关/缺失的带原因，不猜值。"""
        ws = self.wb["Prefill"]
        op_header = next(r for r in range(1, ws.max_row + 1)
                         if ws.cell(r, 1).value == "黑盒标记")
        rows = {ws.cell(r, 2).value: r for r in range(op_header + 1, ws.max_row + 1)
                if ws.cell(r, 2).value}
        flatten = rows["moe_flatten"]
        for col in (17, 18):
            cell = ws.cell(flatten, col)
            self.assertEqual(cell.data_type, "f", (col, cell.value))
            self.assertIn("IF(", str(cell.value))
        self.assertEqual(str(ws.cell(flatten, 19).value), '="（无）"')
        # moe_dispatch 的 N_hit 是数据相关量：实际 shape 明确未知，不写 0。
        dispatch = rows["moe_dispatch"]
        texts = [str(ws.cell(dispatch, col).value) for col in (17, 18, 19)]
        joined = " ".join(texts)
        self.assertIn("N_hit", joined)
        self.assertIn("未知", joined)
        self.assertNotIn("N_hit=0", joined)

    def test_inputs_are_fillable_examples(self):
        for name, expected in (("Prefill", (8, 0)), ("Decode", (1, 4096))):
            ws = self.wb[name]
            rows = _input_rows(ws)
            self.assertEqual(ws.cell(rows["seqlen"], 4).value, expected[0])
            self.assertEqual(ws.cell(rows["kvcachelen"], 4).value, expected[1])
            self.assertEqual(ws.cell(rows["batch_size"], 4).value, 1)
            self.assertEqual(ws.cell(rows["compute_precision"], 4).value, "FP32")
            self.assertEqual(ws.cell(rows["weight_storage_bits"], 4).value, 32)
            self.assertEqual(ws.cell(rows["activation_storage_bits"], 4).value, 32)
            self.assertIn("未知", str(ws.cell(rows["expert_hit"], 4).value))
            for key in ("seqlen", "kvcachelen", "batch_size"):
                self.assertEqual(ws.cell(rows[key], 7).value, "可改")

    def test_derived_values_are_formulas_referencing_inputs_and_own_group(self):
        for name in self.wb.sheetnames:
            ws = self.wb[name]
            inputs = _input_rows(ws)
            headers = [r for r in range(1, ws.max_row + 1)
                       if ws.cell(r, 1).value == "参数id"]
            input_cells = {"D" + str(row) for row in inputs.values()}
            for header in headers:
                rows = _param_rows(ws, header)
                by_id = {ws.cell(r, 1).value: r for r in rows}
                allowed = input_cells | {"D" + str(r) for r in rows}
                for pid in ("B", "seqlen", "kvcachelen"):
                    cell = ws.cell(by_id[pid], 4)
                    self.assertTrue(str(cell.value).startswith("="), (name, pid))
                    targets = {col + str(row) for col, row in _formula_targets(str(cell.value))}
                    self.assertTrue(targets <= input_cells, (name, pid, targets))
                for pid in ("Nwin", "Ncomp", "Lkv", "Lkvis", "Lkvp", "Ncompi", "Kisel", "Bbias"):
                    if pid not in by_id:
                        continue
                    cell = ws.cell(by_id[pid], 4)
                    self.assertTrue(str(cell.value).startswith("="), (name, pid, cell.value))
                    for col, row in _formula_targets(str(cell.value)):
                        self.assertIn(col + str(row), allowed, (name, pid, col, row))
                # 派生行不写死预计算常量：公式单元格不是数值。
                if "Ncomp" in by_id:
                    self.assertEqual(ws.cell(by_id["Ncomp"], 4).data_type, "f")
                if "Lkv" in by_id:      # 整网外围块没有 Lkv 参数行
                    self.assertEqual(ws.cell(by_id["Lkv"], 4).data_type, "f")

    def test_independent_expected_values_match_structure_derivations(self):
        """独立用 scenario_evaluator 复算，核对公式所依据的同一表达式。"""
        expectations = {
            "swa-hash": {
                (8, 0): {"L_kv": 8},
                (1, 4096): {"L_kv": 128},
                (4, 4092): {"L_kv": 131},
            },
            "csa-hash": {
                (8, 0): {"L_kv": 8, "N_win": 2, "N_comp": 2},
                (1, 4096): {"L_kv": 128, "N_win": 0, "N_comp": 1024},
                (4, 4092): {"L_kv": 131, "N_win": 1, "N_comp": 1024},
            },
            "csa-topk": {
                (8, 0): {"L_kv": 8, "N_win": 2, "N_comp": 2},
                (1, 4096): {"L_kv": 128, "N_win": 0, "N_comp": 1024},
                (4, 4092): {"L_kv": 131, "N_win": 1, "N_comp": 1024},
            },
            "hca-topk": {
                (8, 0): {"L_kv": 8, "N_win": 0, "N_comp": 0},
                (1, 4096): {"L_kv": 128, "N_win": 0, "N_comp": 32},
                (4, 4092): {"L_kv": 131, "N_win": 1, "N_comp": 32},
            },
        }
        for variant in self.container["variants"]:
            block = variant["structure"]["block"]
            for (seqlen, kvcachelen), expected in expectations[variant["id"]].items():
                evaluated = {
                    entry["id"]: entry
                    for entry in evaluate_derivations(
                        block["derivations"], block["parameters"],
                        {"seqlen": seqlen, "kvcachelen": kvcachelen})
                }
                for key, value in expected.items():
                    if key not in evaluated:
                        continue
                    self.assertEqual(evaluated[key].get("value"), value,
                                     (variant["id"], (seqlen, kvcachelen), key))

    def test_loop_attribution_and_network_summary(self):
        ws = self.wb["Prefill"]
        op_header = next(r for r in range(1, ws.max_row + 1)
                         if ws.cell(r, 1).value == "黑盒标记")
        first_op = op_header + 1
        self.assertIn("SWA + Hash MoE", ws.cell(first_op, 15).value)
        self.assertIn("×2", ws.cell(first_op, 15).value)
        self.assertIn("[0, 1]", ws.cell(first_op, 15).value)
        # 串行汇总：四组真实层索引/次数 2/1/20/20，合计 43；外围入口/出口各 1 次。
        banner = next(r for r in range(1, ws.max_row + 1)
                      if str(ws.cell(r, 1).value).startswith("整网串行汇总"))
        header = next(r for r in range(banner + 1, ws.max_row + 1)
                      if ws.cell(r, 1).value == "序号")
        self.assertEqual(ws.cell(header, 2).value, "段/层组")
        rows = {ws.cell(r, 2).value: r for r in range(header + 1, header + 7)}
        self.assertEqual(set(rows), {"net-periphery/entry", "swa-hash", "csa-hash",
                                     "hca-topk", "csa-topk", "net-periphery/exit"})
        self.assertEqual([ws.cell(rows[gid], 5).value for gid in
                          ("swa-hash", "csa-hash", "hca-topk", "csa-topk")],
                         [2, 1, 20, 20])
        texts = " ".join(str(c.value) for row in ws.iter_rows() for c in row if c.value)
        self.assertIn("4 组层数合计 43", texts)
        # 单层小计×层数只乘一次：F 引用层组的**单层**小计，G=层数，H=F×G。
        group_banners = {
            gid: next(r for r in range(1, banner)
                      if str(ws.cell(r, 1).value).startswith(f"层组 ")
                      and str(ws.cell(r, 1).value).split("（")[1].startswith(gid + "）"))
            for gid in ("swa-hash", "csa-hash", "hca-topk", "csa-topk")}
        for gid, count in (("swa-hash", 2), ("hca-topk", 20)):
            serial = rows[gid]
            start = group_banners[gid]
            subtotal_row = next(r for r in range(start, banner)
                                if str(ws.cell(r, 1).value).startswith("单层算子小计"))
            total_row = next(r for r in range(start, banner)
                             if str(ws.cell(r, 1).value).startswith("层组总计"))
            self.assertEqual(ws.cell(serial, 6).value, f"=P{subtotal_row}")
            self.assertEqual(ws.cell(serial, 7).value, count)
            self.assertIn(f"$F{serial}*$G{serial}", str(ws.cell(serial, 8).value))
            # 层组总计行本身 = 单层小计×层数，只作阅读核对，不被整网再引用。
            self.assertIn(f"$P${subtotal_row}*{count}",
                          str(ws.cell(total_row, 16).value))
            self.assertTrue(str(ws.cell(total_row, 1).value).startswith("层组总计"))

    def test_costs_are_live_user_rule_formulas_not_zero(self):
        """每个算子行的耗时/工作量都是引用本表维度与吞吐的活公式，不是常数/0。"""
        for name in self.wb.sheetnames:
            ws = self.wb[name]
            inputs = _input_rows(ws)
            cube_cell = f"$D${inputs['cube_mac_per_ms']}"
            vec_cell = f"$D${inputs['vec_elem_per_ms']}"
            op_header = next(r for r in range(1, ws.max_row + 1)
                             if ws.cell(r, 1).value == "黑盒标记")
            row = op_header + 1
            seen = 0
            while row <= ws.max_row and ws.cell(row, 2).value:
                kind = ws.cell(row, 20).value
                cost = ws.cell(row, 16).value
                work = ws.cell(row, 25).value
                if str(kind).startswith("容器"):
                    self.assertEqual(cost, 0)          # 容器明确 0 且写明原因
                elif "整网一次性项" in str(cost):
                    self.assertIn("net_", str(cost))   # 引用行：不写数值成本
                else:
                    self.assertIsInstance(cost, str)
                    self.assertTrue(cost.startswith("="), (name, row, cost))
                    self.assertIn("/", cost)
                    self.assertIn("CUBE" if "CUBE" in kind else "VEC", kind)
                    self.assertIn(cube_cell if "CUBE" in kind else vec_cell, cost)
                    self.assertTrue(work.startswith("="), (name, row, work))
                    refs = ("U", "V", "W", "X") if "CUBE" in kind else ("V", "X")
                    for ref in refs:
                        self.assertIn(f"{ref}{row}", work)
                    seen += 1
                row += 1
            self.assertGreater(seen, 40)
            # 不用 IFERROR(...,0) 把未知折成 0；成本小计也不出现忽略文本的裸 SUM。
            for cell in (c for r in ws.iter_rows() for c in r if c.value):
                text = str(cell.value)
                for fallback in _iferror_fallbacks(text):
                    self.assertTrue(fallback == "FALSE" or fallback.startswith('"'),
                                    (name, cell.coordinate, fallback))
                # 成本/工作量列不得用裸 SUM 忽略文本（串行区“已知部分小计”在贡献列，
                # 已显式标注“不是整网总计”，不在此列）。
                if cell.column in (16, 25) and text.startswith("=SUM("):
                    self.fail(f"不应出现忽略文本的成本 SUM: {cell.coordinate} {text}")
    def test_every_operator_classified_and_mask_rope_charged_once(self):
        """全覆盖：四层组 + 整网外围每个算子都有 CUBE/VEC 规则；mask/RoPE 只收一次。"""
        from structure_reader.export import deepseek_perf_rules as rules

        report = rules.rules_report()
        covered = {(item["scope"], item["operator_id"]) for item in report}
        for variant in self.container["variants"]:
            for op in variant["structure"]["block"]["operators"]:
                self.assertIn((variant["id"], op["id"]), covered)
        for op in self.network["path"]["operators"]:
            self.assertIn(("net-periphery", op["id"]), covered)
        # 每个算子行都有明确分类，没有“未分类”留白。
        ws = self.wb["Prefill"]
        classifications = []
        for header in [r for r in range(1, ws.max_row + 1)
                       if ws.cell(r, 1).value == "黑盒标记"]:
            row = header + 1
            while row <= ws.max_row and ws.cell(row, 2).value:
                classifications.append(str(ws.cell(row, 20).value))
                row += 1
        self.assertTrue(all(kind.startswith(("CUBE", "VEC", "容器"))
                            for kind in classifications), classifications[:5])
        # 层组内的 mask/RoPE 行只作引用；其成本单元格是文字，不进入单层小计区间。
        excluded = [r for r in range(1, ws.max_row + 1)
                    if isinstance(ws.cell(r, 2).value, str)
                    and ws.cell(r, 2).value in ("mask_model_build", "rope_main",
                                                "rope_compress")
                    and "整网一次性项" in str(ws.cell(r, 16).value)]
        self.assertTrue(excluded)
        for row in excluded:
            self.assertIn("不计入本组小计", str(ws.cell(row, 16).value))
        subtotal = next(r for r in range(1, ws.max_row + 1)
                        if str(ws.cell(r, 1).value).startswith("单层算子小计"))
        formula = str(ws.cell(subtotal, 16).value)
        spans = [(int(a), int(b or a))
                 for a, b in re.findall(r"P(\d+)(?::P(\d+))?", formula)]
        self.assertTrue(spans)
        for row in excluded:
            self.assertFalse(any(low <= row <= high for low, high in spans),
                             (row, spans))
        # 整网外围计费：入口段覆盖 net_mask_build / net_rope_main / net_rope_compress。
        entry = [r for r in range(1, ws.max_row + 1)
                 if ws.cell(r, 2).value in ("net_mask_build", "net_rope_main",
                                            "net_rope_compress")
                 and "net_" in str(ws.cell(r, 2).value)]
        self.assertTrue(entry)
        for row in entry:
            self.assertNotIn("整网一次性项", str(ws.cell(row, 16).value))
            self.assertTrue(str(ws.cell(row, 16).value).startswith("="))

    def test_nhit_guard_bounds_topk_and_hash(self):
        """N_hit 相容守卫：≤ min(E, B·seqlen·K)，TopK 非空 ≥ K、Hash ≥ 1。"""
        layer_groups = ("swa-hash", "csa-hash", "hca-topk", "csa-topk")
        ws = self.wb["Prefill"]
        found = {}
        for header in [r for r in range(1, ws.max_row + 1)
                       if ws.cell(r, 1).value == "黑盒标记"]:
            banner = next(r for r in range(header, 0, -1)
                          if str(ws.cell(r, 1).value).startswith("层组 "))
            group = next(gid for gid in (*layer_groups, "net-periphery")
                         if gid in str(ws.cell(banner, 1).value))
            row = header + 1
            while row <= ws.max_row:
                first = ws.cell(row, 1).value
                if isinstance(first, str) and first.startswith(("层组 ", "参数符号区")):
                    break
                if ws.cell(row, 2).value == "moe_expert_slice":
                    found[group] = row
                row += 1
        self.assertEqual(set(found), set(layer_groups))
        for group, row in found.items():
            status = str(ws.cell(row, 26).value)
            note = str(ws.cell(row, 27).value)
            self.assertIn("MIN(", status)
            self.assertIn("N_hit", status)
            self.assertIn("各层统一命中数", note)
            self.assertIn("实际路由不由本工具推断", note)
            if group.endswith("hash"):
                self.assertIn(">=1", status)
            else:
                param_header = max(r for r in range(1, row) if ws.cell(r, 1).value == "参数id")
                k_row = next(r for r in range(param_header + 1, row)
                             if ws.cell(r, 1).value == "K")
                self.assertIn(f">=$D${k_row}", status)

    def test_single_layer_uses_same_exporter_and_declares_examples(self):
        self.assertEqual(self.single_wb.sheetnames, ["Prefill", "Decode"])
        ws = self.single_wb["Prefill"]
        rows = _input_rows(ws)
        self.assertEqual(ws.cell(rows["seqlen"], 4).value, 8)
        self.assertEqual(ws.cell(rows["kvcachelen"], 4).value, 0)
        self.assertEqual(ws.cell(rows["cube_mac_per_ms"], 4).value, 4096)
        self.assertEqual(ws.cell(rows["vec_elem_per_ms"], 4).value, 64)
        self.assertEqual(len([r for r in range(1, ws.max_row + 1)
                              if ws.cell(r, 1).value == "参数id"]), 1)
        texts = " ".join(str(c.value) for row in ws.iter_rows() for c in row if c.value)
        self.assertIn("示例场景", texts)
        self.assertIn("CUBE/VEC", texts)
        # 单层入口没有整网外围块：mask/RoPE 在本层按一次计入，并写明原因。
        self.assertIn("本页没有 net_mask_build 外围块", texts)
        self.assertNotIn(True, [str(ws.cell(r, 1).value).startswith("整网串行汇总")
                                for r in range(1, ws.max_row + 1)])
    def test_unknown_input_keeps_dependents_as_live_unknown_formulas(self):
        """初始未知也必须是活公式：补填输入后应恢复，而不是静态文本。"""
        spec = copy.deepcopy(self.single_spec)
        spec["sheets"][0]["inputs"][0]["value"] = None
        wb = load_workbook(io.BytesIO(export_workbook(spec)))
        ws = wb["Prefill"]
        header = _param_header_row(ws)
        rows = _param_rows(ws, header)
        by_id = {ws.cell(r, 1).value: r for r in rows}
        for pid in ("seqlen", "Lkv", "Nwin", "Ncomp"):
            cell = ws.cell(by_id[pid], 4)
            self.assertEqual(cell.data_type, "f", (pid, cell.value))
            text = str(cell.value)
            self.assertIn("IF(", text)
            self.assertIn("未给定或非法", text)
        self.assertIn("IFERROR", str(ws.cell(by_id["seqlen"], 4).value))
        # 守卫公式不是把未知折成 0。
        self.assertNotEqual(ws.cell(by_id["Lkv"], 4).value, 0)

    def test_axis_symbols_match_parameter_aliases(self):
        """HCA 结构用 Lkv/Ncomp 轴名、参数符号是 L_kv/N_comp；别名必须绑定。"""
        for name in self.wb.sheetnames:
            ws = self.wb[name]
            for row in range(1, ws.max_row + 1):
                value = ws.cell(row, 9).value
                if not isinstance(value, str):
                    continue
                for token in ("Lkv", "Lmask", "Ncomp", "Nwin"):
                    self.assertNotIn(f"未命中本表参数区同名符号：{token}", value,
                                     (name, row, token))
                    self.assertNotIn(f"：{token},", value, (name, row, token))

    def test_group_parameter_scopes_do_not_leak(self):
        for name in self.wb.sheetnames:
            ws = self.wb[name]
            headers = [r for r in range(1, ws.max_row + 1)
                       if ws.cell(r, 1).value == "参数id"]
            ranges = []
            for header in headers:
                rows = _param_rows(ws, header)
                ranges.append((rows[0], rows[-1]))
            for index, header in enumerate(headers):
                rows = _param_rows(ws, header)
                low, high = ranges[index]
                for row in rows:
                    value = ws.cell(row, 4).value
                    if not (isinstance(value, str) and value.startswith("=")):
                        continue
                    for col, target in _formula_targets(value):
                        if col != "D":
                            continue
                        in_group = low <= target <= high
                        in_inputs = target in _input_rows(ws).values()
                        self.assertTrue(in_group or in_inputs,
                                        (name, index, row, target, value))


def _engine_spec(*, extra_repeat_op: bool = False, loop_count: int = 3,
                 cube_default: float = 4096) -> dict:
    """合成 CUBE/VEC fixture：一个 2 ms 的 CUBE 算子、循环 3 层、整网串行汇总。

    B=1、seqlen=1、H=128、N=64 → 工作量 1*1*128*64 = 8192 MAC，÷4096 = 2 ms；
    3 层应得 6 ms（不是 2*3*3=18）。
    """
    params = [
        {"id": "B", "symbol": "B", "meaning": "batch", "value": None,
         "expression": "引用本表 batch_size", "note": "", "source_refs": [],
         "value_ref": "batch_size", "formula": None, "unknown_reason": ""},
        {"id": "seqlen", "symbol": "seqlen", "meaning": "seqlen", "value": None,
         "expression": "引用本表 seqlen", "note": "", "source_refs": [],
         "value_ref": "seqlen", "formula": None, "unknown_reason": ""},
        {"id": "H", "symbol": "H", "meaning": "hidden", "value": 128,
         "expression": None, "note": "", "source_refs": [],
         "value_ref": None, "formula": None, "unknown_reason": ""},
        {"id": "N", "symbol": "N", "meaning": "out", "value": 64,
         "expression": None, "note": "", "source_refs": [],
         "value_ref": None, "formula": None, "unknown_reason": ""},
    ]
    operators = [{
        "id": "op_cube", "module_path": "Fixture.linear", "name": "fixture linear",
        "op": "linear", "inputs": ["t_x"], "outputs": ["t_y"], "weights": [],
        "parameter_refs": ["H", "N"], "source_refs": [],
        "analysis_status": "analyzed", "note": "",
        "performance": {"class": "cube",
                        "terms": [{"label": "Y=X·Wᵀ", "b": "B", "m": "seqlen",
                                   "k": "H", "n": "N"}],
                        "note": "fixture：单次矩阵乘"},
    }]
    if extra_repeat_op:
        params.append({"id": "K", "symbol": "K", "meaning": "重复次数（未给定）",
                       "value": None, "expression": None, "note": "",
                       "source_refs": [], "value_ref": None, "formula": None,
                       "unknown_reason": ""})
        operators.append({
            "id": "op_repeat", "module_path": "Fixture.repeat", "name": "fixture repeat",
            "op": "linear", "inputs": ["t_x"], "outputs": ["t_y"], "weights": [],
            "parameter_refs": ["H", "N", "K"], "source_refs": [],
            "analysis_status": "analyzed", "note": "",
            "performance": {"class": "cube",
                            "terms": [{"label": "重复项", "b": "B", "m": "seqlen",
                                       "k": "H", "n": "N"}],
                            "repeat": "2*K", "note": "fixture：repeat 是符号表达式"},
        })
    return {
        "schema": "operator_workbook.v2",
        "model": {"model_key": "fixture_engine", "display_name": "性能区 fixture",
                  "implementation": "transformers", "source_revision": "fixture",
                  "config_path": "fixtures/config.json"},
        "performance": {
            "status": "fixture 性能区",
            "note": "fixture：用户 CUBE/VEC 规则",
            "engine": {"kind": "cube_vec",
                       "cube_throughput_input": "cube_mac_per_ms",
                       "vec_throughput_input": "vec_elem_per_ms",
                       "cube_default": cube_default, "vec_default": 64,
                       "source": "fixture"},
        },
        "disclaimer": "fixture",
        "sheets": [{
            "id": "single", "name": "Layer1", "title": "性能区 fixture",
            "notes": ["fixture"], "inputs": [
                {"id": "batch_size", "symbol": "B", "label": "batch", "value": 1,
                 "kind": "integer", "source": "fixture", "note": "", "editable": True},
                {"id": "seqlen", "symbol": "seqlen", "label": "seqlen", "value": 1,
                 "kind": "integer", "source": "fixture", "note": "", "editable": True},
                {"id": "kvcachelen", "symbol": "kvcachelen", "label": "history",
                 "value": 0, "kind": "integer", "source": "fixture", "note": "",
                 "editable": True},
                {"id": "cube_mac_per_ms", "symbol": "Tcube", "label": "CUBE 吞吐",
                 "value": cube_default, "kind": "number", "source": "fixture",
                 "note": "", "editable": True},
                {"id": "vec_elem_per_ms", "symbol": "Tvec", "label": "VEC 吞吐",
                 "value": 64, "kind": "number", "source": "fixture", "note": "",
                 "editable": True},
            ],
            "groups": [{
                "id": "g1", "label": "Fixture 组",
                "loop": {"count": loop_count, "label": "真实层组循环",
                         "condition": "同组各层各一次", "note": ""},
                "template_note": "fixture", "scope_note": "fixture",
                "parameters": params,
                "tensors": [
                    {"id": "t_x", "name": "x", "shape": ["B", "seqlen", "H"],
                     "parameter_refs": ["B", "seqlen", "H"], "kind": "activation",
                     "note": ""},
                    {"id": "t_y", "name": "y", "shape": ["B", "seqlen", "N"],
                     "parameter_refs": ["B", "seqlen", "N"], "kind": "activation",
                     "note": ""},
                ],
                "operators": operators,
                "cache_notes": [], "coverage_notes": [],
            }],
            "network": {
                "label": "整网串行汇总（fixture）", "periphery": [],
                "groups": [{"id": "g1", "label": "Fixture 组",
                            "layer_indices": list(range(loop_count)),
                            "count": loop_count}],
                "layer_count": loop_count, "notes": [], "cost_status": "fixture",
                "serial_order": [{"kind": "group", "group": "g1"}],
            },
        }],
    }


class TestPerformanceEngine(unittest.TestCase):
    """合成 fixture 验证 CUBE/VEC 公式、汇总只乘一次、repeat 守卫与小数显示。"""

    def setUp(self):
        self.wb = load_workbook(io.BytesIO(export_workbook(_engine_spec())))
        self.ws = self.wb["Layer1"]

    def _row(self, operator_id: str) -> int:
        header = next(r for r in range(1, self.ws.max_row + 1)
                      if self.ws.cell(r, 1).value == "黑盒标记")
        return next(r for r in range(header + 1, self.ws.max_row + 1)
                    if self.ws.cell(r, 2).value == operator_id)

    def test_rule_math_matches_user_baseline(self):
        from structure_reader.export import deepseek_perf_rules as rules
        from structure_reader.scenario_evaluator import evaluate_int

        def work(rule: dict, env: dict) -> int:
            repeat = rule.get("repeat", 1)
            factor = evaluate_int(repeat, env) if isinstance(repeat, str) else repeat
            total = 0
            for term in rule["terms"]:
                keys = ("b", "m", "k", "n") if rule["class"] == "cube" else ("m", "n")
                product = factor
                for key in keys:
                    product *= evaluate_int(term[key], env)
                total += product
            return total

        cube = {"class": "cube",
                "terms": [{"b": "B", "m": "M", "k": "K", "n": "N", "label": "t"}]}
        vec = {"class": "vec", "terms": [{"m": "M", "n": "N", "label": "t"}]}
        # 用户确认的基准：CUBE B1/M1/K64/N64 = 1 ms；B2/M3 按比例 = 6 ms。
        self.assertEqual(work(cube, {"B": 1, "M": 1, "K": 64, "N": 64}) / 4096, 1.0)
        self.assertEqual(work(cube, {"B": 2, "M": 3, "K": 64, "N": 64}) / 4096, 6.0)
        # VEC M1/N64 = 1 ms；M2/N64 = 2 ms；尺寸不整除 64 不向 64 取整。
        self.assertEqual(work(vec, {"M": 1, "N": 64}) / 64, 1.0)
        self.assertEqual(work(vec, {"M": 2, "N": 64}) / 64, 2.0)
        self.assertEqual(work(vec, {"M": 3, "N": 100}) / 64, 300 / 64)
        # 真实 DeepSeek 规则：q_a_proj B1/S4096/K4096/N1024 = 4,194,304 ms。
        q_a = rules.block_rules("swa-hash")["q_a_proj"]
        env = {"B": 1, "seqlen": 4096, "H": 4096, "Rq": 1024}
        self.assertEqual(work(q_a, env) / 4096, 4194304.0)

    def test_cost_formula_chains_dimensions_and_throughput(self):
        row = self._row("op_cube")
        cost = str(self.ws.cell(row, 16).value)
        work = str(self.ws.cell(row, 25).value)
        inputs = _input_rows(self.ws)
        self.assertIn(f"Y{row}", cost)
        self.assertIn(f"$D${inputs['cube_mac_per_ms']}", cost)
        self.assertIn(f"$D${inputs['cube_mac_per_ms']}>0", cost)
        self.assertIn(f"U{row}*V{row}*W{row}*X{row}", work)
        # 维度列本身引用本组参数/本表输入的活单元格。
        for col in (21, 22, 23, 24):
            self.assertTrue(str(self.ws.cell(row, col).value).startswith("="))
        # 小数显示：1/4096 ms 这类非零小值不能被格式成 0.000。
        self.assertIn("0.000000", self.ws.cell(row, 16).number_format)

    def test_serial_summary_multiplies_layer_count_once(self):
        ws = self.ws
        banner = next(r for r in range(1, ws.max_row + 1)
                      if str(ws.cell(r, 1).value).startswith("整网串行汇总"))
        header = next(r for r in range(banner + 1, ws.max_row + 1)
                      if ws.cell(r, 1).value == "序号")
        serial = header + 1
        subtotal_row = next(r for r in range(1, banner)
                            if str(ws.cell(r, 1).value).startswith("单层算子小计"))
        total_row = next(r for r in range(1, banner)
                         if str(ws.cell(r, 1).value).startswith("层组总计"))
        # F=单层小计、G=3、H=F×G：整网 6 ms，而不是 18 ms。
        self.assertEqual(ws.cell(serial, 6).value, f"=P{subtotal_row}")
        self.assertEqual(ws.cell(serial, 7).value, 3)
        self.assertEqual(ws.cell(serial, 8).value,
                         f'=IF(ISNUMBER($F{serial}),$F{serial}*$G{serial},"未知（单次成本未定）")')
        self.assertEqual(ws.cell(total_row, 16).value,
                         f'=IF(ISNUMBER($P${subtotal_row}),$P${subtotal_row}*3,"未知（单层小计未定）")')
        totals = " ".join(str(ws.cell(r, 8).value) for r in range(serial + 1, serial + 5))
        self.assertIn(f"COUNT(H{serial}:H{serial})", totals)
        self.assertIn(f"SUM(H{serial}:H{serial})", totals)

    def test_repeat_expression_guard_merged(self):
        ws = load_workbook(io.BytesIO(
            export_workbook(_engine_spec(extra_repeat_op=True))))["Layer1"]
        header = next(r for r in range(1, ws.max_row + 1) if ws.cell(r, 1).value == "黑盒标记")
        row = next(r for r in range(header + 1, ws.max_row + 1)
                   if ws.cell(r, 2).value == "op_repeat")
        work = str(ws.cell(row, 25).value)
        # K 参数行的单元格必须出现在守卫里（否则 repeat 文本会 #VALUE!）。
        param_header = next(r for r in range(1, ws.max_row + 1)
                            if ws.cell(r, 1).value == "参数id")
        k_row = next(r for r in range(param_header + 1, ws.max_row + 1)
                     if ws.cell(r, 1).value == "K")
        self.assertIn(f"ISNUMBER($D${k_row})", work)
        self.assertIn(f"2*$D${k_row}", work)
        self.assertIn("未知", str(ws.cell(row, 26).value))

    def test_dimension_cells_require_non_negative_integers(self):
        """维度守卫：符号与最终维度都必须是 ≥0 的整数；否则未知（不是负数/截断值）。"""
        row = self._row("op_cube")
        for col in (21, 22, 23, 24):
            formula = str(self.ws.cell(row, col).value)
            self.assertIn("INT(", formula)
            self.assertIn(">=0", formula)
            self.assertIn("ISNUMBER(", formula)
        # 合法零维度仍然可以算出 0（例如空缓存的压缩条目）：守卫是 >=0，不是 >0。
        formula = str(self.ws.cell(row, 22).value)
        self.assertIn(">=0", formula)
        self.assertNotIn(">0", formula)

    def test_repeat_guard_requires_positive_integer(self):
        ws = load_workbook(io.BytesIO(
            export_workbook(_engine_spec(extra_repeat_op=True))))["Layer1"]
        header = next(r for r in range(1, ws.max_row + 1) if ws.cell(r, 1).value == "黑盒标记")
        row = next(r for r in range(header + 1, ws.max_row + 1)
                   if ws.cell(r, 2).value == "op_repeat")
        work = str(ws.cell(row, 25).value)
        self.assertIn("INT(", work)
        self.assertIn(">=1", work)

    def test_engine_requires_positive_throughput_input(self):
        spec = _engine_spec()
        spec["sheets"][0]["inputs"][3]["value"] = 0
        with self.assertRaises(WorkbookExportError) as caught:
            export_workbook(spec)
        self.assertIn("正数", str(caught.exception))
        spec = _engine_spec()
        spec["sheets"][0]["inputs"] = [
            item for item in spec["sheets"][0]["inputs"]
            if item["id"] != "vec_elem_per_ms"]
        with self.assertRaises(WorkbookExportError) as caught:
            export_workbook(spec)
        self.assertIn("VEC 吞吐输入", str(caught.exception))

    def test_missing_operator_classification_is_named_error(self):
        spec = _engine_spec()
        del spec["sheets"][0]["groups"][0]["operators"][0]["performance"]
        with self.assertRaises(WorkbookExportError) as caught:
            export_workbook(spec)
        self.assertIn("CUBE/VEC 类别与维度", str(caught.exception))

    def test_rule_symbol_outside_group_is_named_error(self):
        spec = _engine_spec()
        spec["sheets"][0]["groups"][0]["operators"][0]["performance"]["terms"][0]["k"] = "Kh"
        with self.assertRaises(WorkbookExportError) as caught:
            export_workbook(spec)
        self.assertIn("Kh", str(caught.exception))




if __name__ == "__main__":
    unittest.main(verbosity=2)
