"""Self-contained tests for the product checker (check_reading_document).

Uses small hand-made fixtures that are clearly synthetic (purpose labelled) and
never reads the real candidate / outputs / coordination state. The fixture is
designed to satisfy every independent semantic expectation; each negative test
mutates exactly one property so the expected issue code is unambiguous.
"""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import check_reading_document as crd  # noqa: E402


def _t(tid, kind, shape, refs=()):
    return {"id": tid, "kind": kind, "name": tid, "shape": list(shape), "parameter_refs": list(refs), "note": ""}


def _o(oid, op, inputs, outputs, weights=(), module="m", name=None, status="analyzed"):
    return {
        "id": oid,
        "module_path": module,
        "name": name or oid,
        "op": op,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "weights": list(weights),
        "parameter_refs": [],
        "source_refs": [],
        "analysis_status": status,
        "note": "",
    }


def _mini_phase(pid, title):
    """A synthetic but fully-connected single-layer phase that satisfies all checks."""
    tensors = [
        _t("streams_in", "activation", ["B", "T", "Nhc", "H"]),
        _t("normed", "activation", ["B", "T", "Nhc×H"]),
        _t("mhc_proj", "activation", ["B", "T", "(2+Nhc)×Nhc"]),
        _t("pre", "activation", ["B", "T", "Nhc"]),
        _t("post", "activation", ["B", "T", "Nhc"]),
        _t("comb", "activation", ["B", "T", "Nhc", "Nhc"]),
        _t("comb_t", "activation", ["B", "T", "Nhc", "Nhc"]),
        _t("q_a", "activation", ["B", "T", "Rq"]),
        _t("q_normed", "activation", ["B", "T", "Rq"]),
        _t("q_head", "activation", ["B", "Nh", "T", "Dh"]),
        _t("i_q", "activation", ["B", "Ih", "T", "Id"]),
        _t("kv_all", "activation", ["B", "1", "L", "Dh"]),
        _t("kv_rep", "activation", ["B", "Nh", "L", "Dh"]),
        _t("qk_logits", "activation", ["B", "Nh", "T", "L"]),
        _t("attn_probs", "activation", ["B", "Nh", "T", "L"]),
        _t("attn_head_out", "activation", ["B", "Nh", "T", "Dh"]),
        _t("attn_block_output", "activation", ["B", "T", "H"]),
        _t("expanded", "activation", ["B", "T", "Nhc", "H"]),
        _t("streams_out", "activation", ["B", "T", "Nhc", "H"]),
        _t("ffn_normed", "activation", ["B", "T", "Nhc×H"]),
        _t("ffn_flat", "activation", ["B×T", "H"]),
        _t("expert_gu", "activation", ["N_e", "2Es"]),
        _t("mlp_out", "activation", ["B", "T", "H"]),
        _t("hc_fn", "weight", ["(2+Nhc)×Nhc", "Nhc×H"]),
        _t("q_weight", "weight", ["Rq", "H"]),
        _t("q_norm_weight", "weight", ["Rq"]),
        _t("q_b_weight", "weight", ["Nh×Dh", "Rq"]),
        _t("i_q_weight", "weight", ["Ih×Id", "Rq"]),
        _t("kv_weight", "weight", ["Dh", "H"]),
        _t("kv_norm_weight", "weight", ["Dh"]),
        _t("o_b_weight", "weight", ["H", "G×Ro"]),
        _t("ffn_norm_weight", "weight", ["H"]),
        _t("expert_gate_up", "weight", ["E", "2Es", "H"]),
    ]
    ops = [
        _o("hc_flatten", "view+float+rmsnorm", ["streams_in"], ["normed"]),
        _o("hc_project", "linear", ["normed"], ["mhc_proj"], weights=["hc_fn"]),
        _o("hc_pre", "affine+sigmoid+add", ["mhc_proj"], ["pre"]),
        _o("hc_comb", "affine+softmax+sinkhorn", ["mhc_proj"], ["post", "comb"]),
        _o("q_a_proj", "linear", ["normed"], ["q_a"], weights=["q_weight"]),
        _o("q_a_norm", "rmsnorm", ["q_a", "q_norm_weight"], ["q_normed"]),
        _o("q_up", "linear+transpose", ["q_normed"], ["q_head"], weights=["q_b_weight"]),
        _o("i_q_proj", "linear+rope", ["q_normed"], ["i_q"], weights=["i_q_weight"]),
        _o("kv_proj", "linear+rmsnorm", ["normed", "kv_norm_weight"], ["kv_all"], weights=["kv_weight"]),
        _o("repeat_kv", "expand+reshape", ["kv_all"], ["kv_rep"]),
        _o("qk", "matmul+mul", ["q_head", "kv_rep"], ["qk_logits"]),
        _o("softmax", "cat+sub+softmax+slice", ["qk_logits"], ["attn_probs"]),
        _o("av", "matmul+transpose", ["attn_probs", "kv_rep"], ["attn_head_out"]),
        _o("o_proj", "linear+flatten", ["attn_head_out"], ["attn_block_output"], weights=["o_b_weight"]),
        _o("attn_expand", "broadcast-mul", ["post", "attn_block_output"], ["expanded"]),
        _o("attn_comb_t", "transpose", ["comb"], ["comb_t"]),
        _o("attn_mix", "matmul+add", ["comb_t", "streams_in", "expanded"], ["streams_out"]),
        _o("ffn_norm", "rmsnorm", ["streams_out", "ffn_norm_weight"], ["ffn_normed"]),
        _o("ffn_flat", "view", ["ffn_normed"], ["ffn_flat"]),
        _o("expert", "linear+chunk+clamp", ["ffn_flat"], ["expert_gu"], weights=["expert_gate_up"]),
        _o("moe_combine", "index_add", ["expert_gu"], ["mlp_out"]),
    ]
    params = [
        {"id": "B", "symbol": "B", "meaning": "batch", "value": None, "expression": None, "note": "", "source_refs": []},
        {"id": "T", "symbol": "T", "meaning": "tokens", "value": None, "expression": None, "note": "", "source_refs": []},
        {"id": "H", "symbol": "H", "meaning": "hidden", "value": 4096, "expression": None, "note": "", "source_refs": []},
        {"id": "Nhc", "symbol": "Nhc", "meaning": "streams", "value": 4, "expression": None, "note": "", "source_refs": []},
    ]
    return {
        "id": pid,
        "title": title,
        "parameters": params,
        "tensors": tensors,
        "operators": ops,
        "cache_notes": ["synthetic fixture"],
        "coverage_notes": ["synthetic fixture"],
    }


def mini_document():
    return {
        "schema": crd.SCHEMA,
        "model": {
            "model_key": "deepseek_v4_flash_0731_hf",
            "display_name": "synthetic",
            "implementation": "transformers",
            "source_revision": "0" * 40,
            "config_path": "models/x/config.json",
        },
        "scope": {
            "layer_index": 4,
            "attention_type": "compressed_sparse_attention",
            "moe_type": "moe",
            "analysis_path": "eager",
            "notes": [],
        },
        "phases": [_mini_phase("prefill", "Prefill"), _mini_phase("decode", "Decode")],
    }


def codes(issues):
    return [i.code for i in issues]


class HappyFixtureTests(unittest.TestCase):
    def test_fixture_has_no_errors_or_manual(self):
        issues = crd.check_reading_document(mini_document())
        errors = [i for i in issues if i.severity == "error"]
        manual = [i for i in issues if i.severity == "manual"]
        self.assertEqual(errors, [], f"unexpected errors: {[i.as_dict() for i in errors]}")
        self.assertEqual(manual, [], f"unexpected manual: {[i.as_dict() for i in manual]}")

    def test_header_and_phase_validation(self):
        doc = mini_document()
        doc["scope"]["layer_index"] = 3
        issues = crd.check_document_header(doc)
        self.assertIn("scope_layer", codes(issues))
        doc = mini_document()
        doc["phases"] = list(reversed(doc["phases"]))
        self.assertIn("phase_order", codes(crd.check_document_header(doc)))


class IntegrityNegativeTests(unittest.TestCase):
    def test_dangling_tensor_ref(self):
        doc = mini_document()
        doc["phases"][0]["operators"][1]["inputs"] = ["does_not_exist"]
        self.assertIn("dangling_tensor_ref", codes(crd.check_reading_document(doc)))

    def test_duplicate_producer_when_not_inplace(self):
        doc = mini_document()
        p = doc["phases"][0]
        p["tensors"].append(_t("comb2", "activation", ["B", "T", "Nhc", "Nhc"]))
        p["operators"].append(_o("comb_again", "affine+softmax+sinkhorn", ["mhc_proj"], ["comb"]))
        self.assertIn("duplicate_producer", codes(crd.check_reading_document(doc)))

    def test_inplace_chain_allowed(self):
        doc = mini_document()
        p = doc["phases"][0]
        p["operators"].append(_o("qk_maskadd", "add", ["qk_logits"], ["qk_logits"]))
        self.assertNotIn("duplicate_producer", codes(crd.check_reading_document(doc)))

    def test_missing_producer(self):
        doc = mini_document()
        p = doc["phases"][0]
        p["operators"] = [o for o in p["operators"] if o["id"] != "kv_proj"]
        self.assertIn("missing_producer", codes(crd.check_reading_document(doc)))

    def test_ordering_violation(self):
        doc = mini_document()
        p = doc["phases"][0]
        ops = p["operators"]
        sm = next(o for o in ops if o["id"] == "softmax")
        ops.remove(sm)
        ops.append(sm)  # softmax now after av
        issues = crd.check_reading_document(doc)
        self.assertIn("ordering", codes(issues))

    def test_unreferenced_tensor_warning(self):
        doc = mini_document()
        doc["phases"][0]["tensors"].append(_t("orphan", "activation", ["B", "T", "H"]))
        issues = crd.check_reading_document(doc)
        self.assertIn("unreferenced_tensor", codes(issues))


class SemanticNegativeTests(unittest.TestCase):
    def test_q_residual_norm_missing_weight(self):
        doc = mini_document()
        p = doc["phases"][0]
        norm = next(o for o in p["operators"] if o["id"] == "q_a_norm")
        norm["inputs"] = ["q_a"]  # drop the learned weight input
        self.assertIn("q_residual_norm_weight", codes(crd.check_reading_document(doc)))

    def test_indexer_query_uses_raw_q(self):
        doc = mini_document()
        p = doc["phases"][0]
        o = next(x for x in p["operators"] if x["id"] == "i_q_proj")
        o["inputs"] = ["q_a"]
        self.assertIn("indexer_query_norm", codes(crd.check_reading_document(doc)))

    def test_av_wrong_input(self):
        doc = mini_document()
        p = doc["phases"][0]
        o = next(x for x in p["operators"] if x["id"] == "av")
        o["inputs"] = ["q_head", "kv_rep"]  # qk logits instead of probabilities
        self.assertIn("av_inputs", codes(crd.check_reading_document(doc)))

    def test_attention_prob_wrong_axis(self):
        doc = mini_document()
        p = doc["phases"][0]
        t = next(x for x in p["tensors"] if x["id"] == "attn_probs")
        t["shape"] = ["B", "Nh", "T", "Nkv"]
        self.assertIn("attn_prob_axis", codes(crd.check_reading_document(doc)))

    def test_transpose_wrong_rank(self):
        doc = mini_document()
        p = doc["phases"][0]
        t = next(x for x in p["tensors"] if x["id"] == "comb_t")
        t["shape"] = ["B", "T", "Nhc", "H"]
        self.assertIn("transpose_rank", codes(crd.check_reading_document(doc)))

    def test_transpose_output_isolated(self):
        doc = mini_document()
        p = doc["phases"][0]
        mix = next(o for o in p["operators"] if o["id"] == "attn_mix")
        mix["inputs"] = ["comb", "streams_in", "expanded"]  # bypass comb_t
        c = codes(crd.check_reading_document(doc))
        self.assertIn("transpose_isolated", c)

    def test_expert_self_contained(self):
        doc = mini_document()
        p = doc["phases"][0]
        e = next(o for o in p["operators"] if o["id"] == "expert")
        e["inputs"] = list(e["outputs"])  # no external routed token
        self.assertIn("expert_input_linkage", codes(crd.check_reading_document(doc)))


class HtmlTests(unittest.TestCase):
    def test_html_dangling_href(self):
        html = '<section class="phase" id="phase-prefill"></section><a href="#missing">x</a>'
        issues = crd.check_html(html, mini_document())
        self.assertIn("html_dangling_href", codes(issues))

    def test_html_phase_order(self):
        html = '<section class="phase" id="phase-decode"></section><section class="phase" id="phase-prefill"></section>'
        self.assertIn("html_phase_order", codes(crd.check_html(html, mini_document())))

    def test_html_blackbox_marker_required(self):
        doc = mini_document()
        doc["phases"][0]["operators"][3]["analysis_status"] = "internal_unanalyzed"
        html = '<section class="phase" id="phase-prefill"></section><section class="phase" id="phase-decode"></section>'
        self.assertIn("html_blackbox_marker", codes(crd.check_html(html, doc)))

    def test_html_clean(self):
        html = '<section class="phase" id="phase-prefill"></section><section class="phase" id="phase-decode"></section>'
        self.assertEqual(crd.check_html(html, mini_document()), [])


class WorkbookTests(unittest.TestCase):
    def _write_workbook(self, doc, path, corrupt_ref=False):
        import openpyxl

        wb = openpyxl.Workbook()
        wb.remove(wb.active)
        for pid, sheet in (("prefill", "Prefill"), ("decode", "Decode")):
            ws = wb.create_sheet(sheet)
            phase = next(p for p in doc["phases"] if p["id"] == pid)
            ws.append(["stage", pid])
            ws.append(["参数id", "符号", "含义", "值", "值类型", "表达式", "来源类型", "来源定位", "说明"])
            for pp in phase["parameters"]:
                ws.append([pp["id"], pp["symbol"], pp["meaning"], "null", "null", "", "", "", ""])
            ws.append([None] * 9)
            ws.append(["算子区"])
            ws.append(["黑盒标记", "算子id", "模块路径", "名称", "算子类型", "输入 名称[shape]", "输出 名称[shape]", "权重 名称[shape]", "参数引用 本表id"])
            tmap = {t["id"]: t for t in phase["tensors"]}
            for o in phase["operators"]:
                def render(ids):
                    return ", ".join(f"{i}[{', '.join(tmap[i]['shape'])}]" for i in ids)
                pref = o["parameter_refs"] or (["BOGUS"] if corrupt_ref else ["Nhc"])
                ws.append(["已展开", o["id"], o["module_path"], o["name"], o["op"], render(o["inputs"]), render(o["outputs"]), render(o["weights"]), ", ".join(pref)])
        wb.save(path)

    def test_workbook_clean(self):
        doc = mini_document()
        for p in doc["phases"]:
            p["parameters"] = p["parameters"] + [
                {"id": "Rq", "symbol": "Rq", "meaning": "", "value": None, "expression": None, "note": "", "source_refs": []}
            ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "wb.xlsx"
            self._write_workbook(doc, path)
            self.assertEqual([i for i in crd.check_workbook(path, doc) if i.severity == "error"], [])

    def test_workbook_dangling_param_link(self):
        doc = mini_document()
        doc["phases"][0]["operators"][0]["parameter_refs"] = ["NOT_A_PARAM"]
        for p in doc["phases"]:
            p["operators"][0]["parameter_refs"] = ["NOT_A_PARAM"]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "wb.xlsx"
            self._write_workbook(doc, path)
            self.assertIn("xlsx_param_link", codes(crd.check_workbook(path, doc)))

    def test_workbook_wrong_sheet_order(self):
        import openpyxl

        doc = mini_document()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "wb.xlsx"
            wb = openpyxl.Workbook()
            wb.remove(wb.active)
            wb.create_sheet("Decode")
            wb.create_sheet("Prefill")
            wb.save(path)
            self.assertIn("xlsx_sheets", codes(crd.check_workbook(path, doc)))


if __name__ == "__main__":
    unittest.main()
