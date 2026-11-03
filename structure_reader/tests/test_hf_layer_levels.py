"""T03：一份参数化 DeepSeek Block 结构 + seqlen/kvcachelen 场景 + 分层浏览页。

这些测试针对真实语义边界（不是实现镜像）：
- 同一个层组只构建**一份**结构；prefill/decode 是同一结构的场景投影；
- seqlen=本次新 token 数、kvcachelen=调用前逻辑历史数（空缓存 Prefill 为 0），
  结构里保留符号；派生量按固定配置 + 连续执行前提推导，未知不默认 0/False；
- 折叠层级不丢真实依赖（残差/缓存读写/共享 KV/压缩窗口/Indexer 条偏置）；
- 页面默认只显示当前层级，进入/返回链接有界，未知场景项如实标注。
"""
from __future__ import annotations

import copy
import hashlib
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parent
REPO_ROOT = WS

_READY = (WS / "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json").is_file()
_skip = unittest.skipUnless(_READY, "缺少只读 HF 输入；如实跳过")


def _derivations(structure: dict, scenario: str) -> dict:
    return {row["id"]: row
            for row in structure["scenario_derivations"][scenario]}


@_skip
class TestOneParameterizedStructure(unittest.TestCase):
    """每个真实层组只有一份结构，四类层型都能构建。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import (
            build_layer_documents, build_layer_structure)

        cls.build = staticmethod(build_layer_structure)
        cls.build_docs = staticmethod(build_layer_documents)
        cls.structures = {idx: build_layer_structure(REPO_ROOT, layer_index=idx)
                          for idx in (0, 2, 3, 4)}

    def test_schema_and_single_block(self):
        for idx, structure in self.structures.items():
            self.assertEqual(structure["schema"], "hf_layer_structure.v1")
            self.assertNotIn("phases", structure,
                             "结构文档不能按 prefill/decode 复制两套")
            block = structure["block"]
            self.assertEqual(block["id"], "block")
            self.assertTrue(block["operators"] and block["tensors"])
            self.assertTrue(block["cache_notes"])
            self.assertTrue(block["coverage_notes"])
            self.assertGreaterEqual(len(block["derivations"]), 3)

    def test_scenario_parameters_are_symbolic_and_named_by_domain(self):
        for idx, structure in self.structures.items():
            params = {p["id"]: p for p in structure["block"]["parameters"]}
            for pid in ("seqlen", "kvcachelen"):
                self.assertIn(pid, params, f"层 {idx} 缺少 {pid}")
                self.assertIsNone(params[pid]["value"],
                                  f"{pid} 在结构里必须保留符号（场景另行绑定）")
            seq = params["seqlen"]["meaning"] + params["seqlen"]["note"]
            kv = params["kvcachelen"]["meaning"] + params["kvcachelen"]["note"]
            self.assertIn("新 token", seq)
            self.assertIn("不含", seq)
            self.assertIn("逻辑历史", kv)
            self.assertIn("空缓存", kv)
            self.assertIn("0", kv)
            self.assertTrue("不是缓存字节数" in kv)
            self.assertIn("不等于", kv)

    def test_no_hidden_phase_symbols_or_plus_one(self):
        """旧 T/P 轴与 Decode 固定 +1/T=1 的写法不得残留在结构事实里。

        source_refs 里引用源码原文（如 ``seq_len==1 or compressed_len==0``）是证据，
        不算结构假设；这里只看参数/张量/算子的说明与表达式。
        """
        for idx, structure in self.structures.items():
            block = structure["block"]
            shape_blob = repr([t["shape"] for t in block["tensors"]])
            self.assertNotIn("'T'", shape_blob, f"层 {idx} shape 仍用旧符号 T")
            self.assertNotIn("'P'", shape_blob, f"层 {idx} shape 仍用旧符号 P")
            facts = []
            for item in (*block["parameters"], *block["tensors"], *block["operators"]):
                facts.append(str(item.get("meaning", "")))
                facts.append(str(item.get("expression", "")))
                facts.append(str(item.get("note", "")))
            facts.extend(str(d["expression"]) + str(d["note"])
                         for d in block["derivations"])
            blob = " ".join(facts)
            for bad in ("L_kv=min(P,W-1)+1", "T=1", "P=0"):
                self.assertNotIn(bad, blob, f"层 {idx} 残留阶段固定假设 {bad}")
            # 关键规模公式必须用统一场景参数
            self.assertIn("kvcachelen", blob)
            self.assertIn("seqlen", blob)

    def test_contract_validator_accepts_real_structures(self):
        from structure_reader.checks.deepseek_v4_hf_contracts import (
            validate_hf_layer_structure)

        for idx, structure in self.structures.items():
            self.assertEqual(validate_hf_layer_structure(structure), [],
                             f"层 {idx} 结构契约不应有问题")

    def test_legacy_projection_shares_structure_facts(self):
        """prefill/decode 投影只换场景取值，张量/算子/形状完全同源。"""
        _, document = self.build_docs(REPO_ROOT, layer_index=4)
        self.assertEqual([p["id"] for p in document["phases"]], ["prefill", "decode"])
        prefill, decode = document["phases"]
        self.assertEqual([t["id"] for t in prefill["tensors"]],
                         [t["id"] for t in decode["tensors"]])
        self.assertEqual([o["id"] for o in prefill["operators"]],
                         [o["id"] for o in decode["operators"]])
        self.assertEqual([t["shape"] for t in prefill["tensors"]],
                         [t["shape"] for t in decode["tensors"]])
        self.assertEqual([t.get("kind") for t in prefill["tensors"]],
                         [t.get("kind") for t in decode["tensors"]])
        kv_p = {p["id"]: p for p in prefill["parameters"]}["kvcachelen"]
        kv_d = {p["id"]: p for p in decode["parameters"]}["kvcachelen"]
        seq_p = {p["id"]: p for p in prefill["parameters"]}["seqlen"]
        seq_d = {p["id"]: p for p in decode["parameters"]}["seqlen"]
        self.assertEqual(kv_p["value"], 0)
        self.assertIsNone(kv_d["value"])
        self.assertIsNone(seq_p["value"])
        self.assertEqual(seq_d["value"], 1, "Decode 单 token 只是场景预设")


@_skip
class TestScenarioDerivations(unittest.TestCase):
    """少量有区别的场景：空历史 Prefill、非零历史单 token、闭窗/未闭窗、多 token。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_layer_structure

        cls.csa = build_layer_structure(REPO_ROOT, layer_index=4)
        cls.hca = build_layer_structure(REPO_ROOT, layer_index=3)
        cls.swa = build_layer_structure(REPO_ROOT, layer_index=0)
        cls.ids = {structure["scope"]["default_scenario"] for structure in
                   (cls.csa, cls.hca, cls.swa)}

    def test_default_scenario_is_empty_cache_prefill(self):
        self.assertEqual(self.ids, {"prefill-empty"})
        rows = _derivations(self.csa, "prefill-empty")
        self.assertEqual(rows["L_kv"]["value"], 8)
        self.assertEqual(rows["Tbuf"]["value"], 0, "空缓存 Prefill 的 kvcachelen=0")
        self.assertEqual(rows["N_win"]["value"], 2)
        self.assertEqual(rows["N_comp"]["value"], 2)

    def test_nonzero_history_single_token_decode_open_window(self):
        rows = _derivations(self.csa, "decode-nonzero")
        self.assertEqual(rows["L_kv"]["value"], 128,
                         "min(kvcachelen, W-1)+seqlen = 127+1")
        self.assertEqual(rows["L_kv_post"]["value"], 127)
        self.assertEqual(rows["N_win"]["value"], 0, "4096 不与 M=4 对齐：本调用不闭窗")
        self.assertEqual(rows["N_comp"]["value"], 1024)
        self.assertEqual(rows["first_window_position"]["value"], 4096)
        self.assertEqual(rows["future_threshold"]["value"], 1024)
        self.assertEqual(rows["Ki_sel"]["value"], 512)
        # 物理保留长度不是逻辑历史长度
        self.assertNotEqual(rows["L_kv"]["value"], 4096)

    def test_window_closed_scenarios(self):
        closed = _derivations(self.csa, "decode-closed")
        self.assertEqual(closed["Tbuf"]["value"], 3)
        self.assertEqual(closed["N_win"]["value"], 1)
        self.assertEqual(closed["Tbuf_post"]["value"], 0)
        self.assertEqual(closed["first_window_position"]["value"], 4092)

        multi = _derivations(self.csa, "decode-multi")
        self.assertEqual(multi["L_kv"]["value"], 131, "min(4092,127)+4")
        self.assertEqual(multi["N_win"]["value"], 1, "seqlen=4 也按 (Tbuf+seqlen)//M")
        self.assertEqual(multi["N_comp"]["value"], 1024)
        self.assertEqual(multi["Tbuf_post"]["value"], 0)

    def test_hca_uses_its_own_window_width_and_block_bias_condition(self):
        self.assertEqual({p["id"]: p for p in self.hca["block"]["parameters"]}["M"]["value"], 128)
        openw = _derivations(self.hca, "decode-nonzero")
        self.assertEqual(openw["N_win"]["value"], 0)
        self.assertEqual(openw["N_comp"]["value"], 32)
        self.assertIs(openw["block_bias"]["value"], False,
                      "seqlen=1 → 源码返回 None（不物化）")
        multi = _derivations(self.hca, "decode-multi")
        self.assertEqual(multi["N_win"]["value"], 1)
        self.assertIs(multi["block_bias"]["value"], True,
                      "seqlen=4 且条目>0 → 物化")
        empty = _derivations(self.hca, "prefill-empty")
        self.assertIs(empty["block_bias"]["value"], False,
                      "seqlen=8 但本调用无条目 → 不物化")

    def test_swa_has_no_compression_derivations(self):
        rows = _derivations(self.swa, "decode-nonzero")
        self.assertEqual(set(rows), {"L_kv", "L_mask", "L_kv_post"})
        self.assertEqual(rows["L_kv"]["value"], 128)

    def test_symbolic_scenario_is_honestly_unknown_and_propagates(self):
        rows = _derivations(self.csa, "prefill-symbolic")
        self.assertEqual(rows["L_kv"]["state"], "unknown")
        self.assertIn("seqlen", rows["L_kv"]["unknown_reason"])
        self.assertEqual(rows["Tbuf"]["value"], 0, "kvcachelen=0 已知，仍可确定")
        # 依赖未知量的派生量也未知，不默认成 0
        self.assertEqual(rows["N_comp"]["state"], "unknown")
        self.assertIn("N_win", rows["N_comp"]["unknown_reason"])
        self.assertEqual(rows["Ki_sel"]["state"], "unknown")
        self.assertIn("N_comp_i", rows["Ki_sel"]["unknown_reason"])

    def test_causal_threshold_boundary_is_strict(self):
        """源码屏蔽 w >= (t+1)//M：条目恰等于阈值时必须屏蔽（含等于）。"""
        # 官方语义（modeling:437-439 / 574-581）：masked_fill(entry >= threshold, -inf)
        def visible(w, t, m):
            return w < (t + 1) // m

        # 复核给出的边界例：HCA kvcachelen=4092、seqlen=4、M=128
        # 首查询 t=4092 → 阈值 31；条目 31 覆盖 3968..4095，含未来 token，必须屏蔽
        self.assertEqual((4092 + 1) // 128, 31)
        self.assertFalse(visible(31, 4092, 128))
        self.assertTrue(visible(30, 4092, 128))
        # 末查询 t=4095 → 阈值 32；条目 31 已完整，可见
        self.assertEqual((4095 + 1) // 128, 32)
        self.assertTrue(visible(31, 4095, 128))
        self.assertFalse(visible(32, 4095, 128))

        rows = _derivations(self.hca, "decode-multi")
        self.assertEqual(rows["first_query_threshold"]["value"], 31)
        self.assertEqual(rows["future_threshold"]["value"], 32)
        csa = _derivations(self.csa, "prefill-empty")
        self.assertEqual(csa["first_query_threshold"]["value"], 0)
        self.assertEqual(csa["future_threshold"]["value"], 2)

        # 结构与说明必须写严格条件，不能出现 w ≤
        for structure in (self.csa, self.hca):
            block = structure["block"]
            facts = " ".join(
                str(item.get(k, "")) for item in
                (*block["parameters"], *block["tensors"], *block["operators"],
                 *block["derivations"])
                for k in ("meaning", "expression", "note"))
            self.assertNotIn("w ≤", facts)
            self.assertNotIn("w≤", facts)
            self.assertIn("w < (t+1)//M", facts)
            self.assertIn("等于阈值", facts)

    def test_derivative_expressions_are_whitelisted_and_evaluable(self):
        from structure_reader.scenario_evaluator import check_expression

        for structure in (self.csa, self.hca, self.swa):
            for item in structure["block"]["derivations"]:
                check_expression(item["expression"])


@_skip
class TestCollapsedLevelsKeepDependencies(unittest.TestCase):
    """折叠展示不删除真实计算/关系：残差、缓存读写、共享 KV、窗口闭合依赖仍在。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_layer_structure

        cls.s = build_layer_structure(REPO_ROOT, layer_index=4)
        cls.ops = {o["id"]: o for o in cls.s["block"]["operators"]}
        cls.tensors = {t["id"]: t for t in cls.s["block"]["tensors"]}

    def test_residual_mixing_retained_with_comb_transpose(self):
        for seg in ("attn", "ffn"):
            self.assertIn(f"{seg}_residual_combT", self.ops)
            mix = self.ops[f"{seg}_residual_mix"]
            blob = str(mix["note"]) + str(mix["name"])
            self.assertIn("comb", blob)
            self.assertIn(f"{seg}_combT", mix["inputs"])
            self.assertIn(f"{seg}_expanded", mix["inputs"])

    def test_cache_read_before_write_is_two_tensors(self):
        upd = self.ops["cache_update"]
        self.assertIn("cache_keys", upd["inputs"], "读调用前持久态")
        self.assertIn("cache_keys_post", upd["outputs"], "写调用后持久态")
        self.assertIn("cache_keys", self.tensors)
        self.assertIn("cache_keys_post", self.tensors)
        self.assertIn("min(kvcachelen,W-1)", self.tensors["cache_keys"]["shape"][2])
        self.assertIn("min(kvcachelen+seqlen,W-1)",
                      self.tensors["cache_keys_post"]["shape"][2])

    def test_shared_kv_and_compressed_concat_order(self):
        self.assertEqual(self.ops["kv_proj"]["outputs"], ["kv_proj_out"])
        self.assertEqual(self.ops["cache_update"]["inputs"][0], "kv_rope")
        order = [o["id"] for o in self.s["block"]["operators"]]
        self.assertLess(order.index("cache_update"), order.index("kv_concat"))
        self.assertIn("cache_compressed_c_post", self.ops["kv_concat"]["inputs"])
        self.assertIn("kv_full", self.ops["kv_concat"]["inputs"])

    def test_compressor_overlap_and_indexer_dependencies(self):
        win = self.ops["c_window_view"]
        self.assertIn("cache_overlap_c_kv", win["inputs"])
        self.assertIn("cache_overlap_c_kv_post", win["outputs"])
        iwin = self.ops["i_window_view"]
        self.assertIn("cache_overlap_i_kv", iwin["inputs"])
        self.assertIn("i_topk", self.ops)
        self.assertIn("-1", self.ops["i_topk"]["note"])
        self.assertIn("indexer_topk", self.ops["c_block_bias"]["inputs"])

    def test_block_bias_and_mask_extension_retained(self):
        self.assertIn("mask_extend", self.ops)
        self.assertIn("c_block_bias", self.ops["mask_extend"]["inputs"])
        self.assertIn("mask_model", self.ops["mask_extend"]["inputs"])

    def test_expert_template_single_copy(self):
        ids = [o["id"] for o in self.s["block"]["operators"]]
        self.assertEqual(sum(1 for i in ids if i.startswith("moe_expert_")), 6)
        self.assertIn("expert_gate_up_e", self.ops["moe_expert_gu"]["weights"])

    def test_flow_closure_and_unique_ids(self):
        from structure_reader.checks.deepseek_v4_hf_contracts import (
            validate_hf_layer_structure)

        issues = validate_hf_layer_structure(self.s)
        self.assertEqual([i for i in issues if "数据流" in i or "引用" in i or "重复" in i], [])


@_skip
class TestStructurePageLevels(unittest.TestCase):
    """分层浏览页：默认仅当前层级、四级路由、场景面板数据与按需详情。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_network import build_network_overview
        from structure_reader.render.hf_structure_page import render_structure_page
        from structure_reader.extract.hf_layer_structures import build_structures_container
        from structure_reader.rules.deepseek_v4_hf import SOURCE_FILES

        cls.container = build_structures_container(REPO_ROOT)
        cls.items = [
            {key: v[key] for key in ("id", "label", "attention_type", "moe_type",
                                     "layer_indices", "representative_layer_index",
                                     "structure")}
            for v in cls.container["variants"]
        ]
        cls.network = build_network_overview(REPO_ROOT)
        snapshot = {}
        for relpath in SOURCE_FILES.values():
            path = REPO_ROOT / relpath
            if path.suffix in (".py", ".json"):
                snapshot[relpath] = path.read_text(encoding="utf-8")
        cls.shas = {key: hashlib.sha256((REPO_ROOT / relpath).read_bytes()).hexdigest()
                    for key, relpath in SOURCE_FILES.items()}
        cls.html = render_structure_page(
            cls.network, cls.items, source_snapshot=snapshot,
            input_shas=cls.shas).decode("utf-8")

    def test_default_level_is_network_only(self):
        h = self.html
        self.assertIn("#lv-network{display:block}", h.replace("\n", ""))
        self.assertIn(".lv{display:none", h.replace("\n", ""))
        # 其余层级由 JS 路由显示；HTML 里各层级真实存在
        for gid in ("swa-hash", "csa-hash", "hca-topk", "csa-topk"):
            for level in ("group", "block", "ops"):
                self.assertIn(f'id="lv-{level}-{gid}"', h)
        self.assertIn('data-level', h)

    def test_four_levels_are_addressable_and_reversible(self):
        h = self.html
        self.assertIn('href="structure.html#/network"', h)
        for gid in ("swa-hash", "csa-hash", "hca-topk", "csa-topk"):
            self.assertIn(f'href="structure.html#/group/{gid}"', h)
            self.assertIn(f'href="structure.html#/group/{gid}/block"', h)
            self.assertIn(f'href="structure.html#/group/{gid}/ops"', h)
        # 面包屑/返回链接（静态、不依赖脚本）
        self.assertIn("当前位置", h)
        self.assertIn("↑ 返回整网", h)
        self.assertIn("↑ 返回 Block", h)

    def test_group_level_shows_real_membership_and_template(self):
        h = self.html
        for group in self.network["groups"]:
            gid = group["id"]
            section = h[h.index(f'id="lv-group-{gid}"'):]
            section = section[:section.index("</section>")]
            self.assertIn(str(group["representative_layer_index"]), section)
            for index in group["layer_indices"][:3]:
                self.assertIn(str(index), section)
            self.assertIn("同组共用一份模板", section)

    def test_scenario_panel_data_is_embedded_and_honest(self):
        import json

        m = re.search(r'<script id="scn-data" type="application/json">(.*?)</script>',
                      self.html, re.S)
        self.assertIsNotNone(m)
        data = json.loads(m.group(1))
        self.assertEqual(data["default"], "prefill-empty")
        ids = [s["id"] for s in data["scenarios"]]
        for expected in ("prefill-empty", "prefill-symbolic", "decode-nonzero",
                         "decode-closed", "decode-multi"):
            self.assertIn(expected, ids)
        csa = data["groups"]["csa-topk"]["derivations"]
        self.assertEqual(csa["decode-nonzero"][0]["value"], 128)
        unknown = [row for row in csa["prefill-symbolic"] if row["state"] != "determined"]
        self.assertTrue(unknown)
        self.assertTrue(all("unknown_reason" in row for row in unknown))
        # 场景面板由页面脚本渲染，未知项有独立样式（不写成 0）
        self.assertIn("未知（", self.html)
        self.assertIn("class=\"unknown\"", self.html)

    def test_operator_cards_and_sources_are_on_demand(self):
        h = self.html
        # 算子卡默认在折叠的模块内；源码面板初始隐藏（点击展开）
        self.assertIn("details class=\"fold mod\"", h)
        self.assertIn("style=\"display:none\"", h)
        self.assertIn("toggleSource", h)
        self.assertIn("locateInDiagram", h)
        self.assertGreater(h.count('class="src-panel"'), 100)

    def test_no_duplicate_ids_and_offline(self):
        from html.parser import HTMLParser

        class Ids(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []

            def handle_starttag(self, tag, attrs):
                a = dict(attrs)
                if "id" in a:
                    self.ids.append(a["id"])

        parser = Ids()
        parser.feed(self.html)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        self.assertNotRegex(self.html, r'<script[^>]+src="https?://')
        self.assertNotRegex(self.html, r'<link[^>]+href="https?://')

    def test_onnx_links_only_when_assets_exist(self):
        """可选 ONNX/Netron 联查：未启用时不出现；提供 bundle 时才生成真实链接。"""
        from structure_reader.render.hf_structure_page import _onnx_links

        self.assertEqual(_onnx_links("csa-topk", "res_attn", None), "")
        self.assertEqual(_onnx_links("csa-topk", "res_attn", {}), "")
        self.assertNotIn("llmarchi-netron://", self.html)
        bundle = {
            "candidate_id": "cand123",
            "bundles": {"csa-topk": {"prefill": {"res_attn": {
                "base": "onnx/csa-topk/prefill/res_attn",
                "state": {"node_count_original": 12, "node_count_simplified": 9}}}}},
        }
        out = _onnx_links("csa-topk", "res_attn", bundle)
        self.assertIn("llmarchi-netron://open?candidate=cand123", out)
        self.assertIn("phase=prefill", out)
        self.assertIn("group=csa-topk", out)
        self.assertIn("onnx/csa-topk/prefill/res_attn/model.sim.onnx", out)
        self.assertIn("12→9", out)
        self.assertEqual(_onnx_links("csa-topk", "no_such_module", bundle), "")

    def test_structure_construction_lives_in_extract(self):
        """复核 P2 + T04：结构构造归 extract，Excel 导出归 export，render 不再参与。"""
        import importlib.util
        import inspect

        import structure_reader.extract.hf_layer_structures as extract_mod
        import structure_reader.export.hf_workbook as hf_workbook_mod
        import structure_reader.export.operator_workbook as export_mod

        self.assertTrue(hasattr(extract_mod, "build_structures_container"))
        self.assertIn("build_layer_documents", inspect.getsource(extract_mod))
        # T04：分组两表导出已从 render 删除；单层/分组共用 export 的通用导出器。
        self.assertIsNone(importlib.util.find_spec("structure_reader.render.m2_variants"),
                          "render.m2_variants 的重复导出流程应在 T04 删除")
        self.assertTrue(hasattr(export_mod, "export_workbook"))
        self.assertTrue(hasattr(hf_workbook_mod, "build_structures_workbook_spec"))
        self.assertTrue(hasattr(hf_workbook_mod, "export_structures_workbook"))

    def test_anchor_links_are_routable_to_their_owning_level(self):
        """普通锚点（#p/#t/#op/#figop/#mod）必须能推断所属层级，不能被打回整网。

        复核 P2：参数/张量/图联查曾因 parseRoute 只认 ``#/group`` 而隐藏目标。
        这里检查每个内部锚点目标真实存在且位于某个 ``[data-lv]`` section 内
        （兜底路由可判定），并检查路由脚本包含锚点解析/展开/高亮实现。
        """
        h = self.html
        id_pos = {m.group(1): m.start() for m in re.finditer(r'id="([^"]+)"', h)}
        sections = [(m.start(), m.end()) for m in
                    re.finditer(r'<section\b[^>]*data-lv="[^"]+"[^>]*>', h)]
        closes = [m.start() for m in re.finditer(r"</section>", h)]
        spans = []
        for start in (s for s, _ in sections):
            end = next((c for c in closes if c > start), len(h))
            spans.append((start, end))

        inline = set(re.findall(r'href="#([^"#][^"]*)"', h))
        inline = {a for a in inline if not a.startswith("/")}
        self.assertGreater(len(inline), 50, "应保留既有 #p/#t/#op/#figop 联查")
        for anchor in sorted(inline):
            self.assertIn(anchor, id_pos, f"锚点目标缺失: {anchor}")
            pos = id_pos[anchor]
            self.assertTrue(any(s <= pos < e for s, e in spans),
                            f"锚点 {anchor} 不在任何层级 section 内")
        for marker in ("function anchorRoute", "function reveal", ".anchor-target",
                       "location.hash"):
            self.assertIn(marker, h)
        # 目标所属 section 的层级可由 id 前缀直接判定（Block 参数 / 算子层张量）
        self.assertIn('id="lv-block-csa-topk"', h)
        self.assertIn('id="lv-ops-csa-topk"', h)
        self.assertIn('id="block-params-csa-topk"', h)
        block_span = next((s, e) for s, e in spans
                          if f'id="lv-block-csa-topk"' in h[s:h.index(">", s) + 1])
        ops_span = next((s, e) for s, e in spans
                        if f'id="lv-ops-csa-topk"' in h[s:h.index(">", s) + 1])
        for anchor in ("p-block-csa-topk-Nhc", "block-params-csa-topk"):
            self.assertTrue(block_span[0] <= id_pos[anchor] < block_span[1],
                            f"{anchor} 应在 Block 层 section 内")
        self.assertTrue(ops_span[0] <= id_pos["t-block-csa-topk-hidden_streams_in"]
                        < ops_span[1], "张量目标应在算子层 section 内")

    def test_page_rejects_inconsistent_inputs(self):
        from structure_reader.render.hf_structure_page import (
            StructurePageError, render_structure_page)

        with self.assertRaises(StructurePageError):
            render_structure_page(self.network, [])
        with self.assertRaises(StructurePageError):
            render_structure_page(self.network, self.items[:-1])
        bad = copy.deepcopy(self.items)
        bad[0]["structure"] = {"schema": "not_a_structure"}
        with self.assertRaises(StructurePageError):
            render_structure_page(self.network, bad)


@_skip
class TestStructureContractGates(unittest.TestCase):
    """结构契约门禁：破坏一份结构/场景 honesty 必须被抓住。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_layer_structure

        cls.doc = build_layer_structure(REPO_ROOT, layer_index=4)
        from structure_reader.checks.deepseek_v4_hf_contracts import (
            validate_hf_layer_structure)
        cls.validate = staticmethod(validate_hf_layer_structure)

    def test_phases_duplicate_rejected(self):
        bad = copy.deepcopy(self.doc)
        bad["phases"] = [{"id": "prefill"}, {"id": "decode"}]
        self.assertTrue(any("phases" in i for i in self.validate(bad)))

    def test_missing_scenario_parameter_rejected(self):
        bad = copy.deepcopy(self.doc)
        bad["block"]["parameters"] = [
            p for p in bad["block"]["parameters"] if p["id"] != "kvcachelen"]
        self.assertTrue(any("kvcachelen" in i for i in self.validate(bad)))

    def test_bound_scenario_value_rejected(self):
        bad = copy.deepcopy(self.doc)
        for p in bad["block"]["parameters"]:
            if p["id"] == "seqlen":
                p["value"] = 8
        self.assertTrue(any("保留符号" in i for i in self.validate(bad)))

    def test_legacy_axis_residue_rejected(self):
        bad = copy.deepcopy(self.doc)
        bad["block"]["tensors"][0]["shape"] = ["B", "T"]
        self.assertTrue(any("旧符号" in i for i in self.validate(bad)))

    def test_invalid_derivation_expression_rejected(self):
        bad = copy.deepcopy(self.doc)
        bad["block"]["derivations"][0]["expression"] = "import os"
        self.assertTrue(any("表达式非法" in i for i in self.validate(bad)))

    def test_symbolic_scenario_without_unknown_rejected(self):
        bad = copy.deepcopy(self.doc)
        for row in bad["scenario_derivations"]["prefill-symbolic"]:
            row["state"] = "determined"
            row["value"] = 0
            row.pop("unknown_reason", None)
        self.assertTrue(any("不诚实" in i for i in self.validate(bad)))

    def test_concrete_scenario_with_unknown_rejected(self):
        bad = copy.deepcopy(self.doc)
        bad["scenario_derivations"]["decode-nonzero"][0]["state"] = "unknown"
        bad["scenario_derivations"]["decode-nonzero"][0]["unknown_reason"] = "缺少变量: seqlen"
        self.assertTrue(any("仍有未知派生量" in i for i in self.validate(bad)))

    def test_threshold_boundary_reverted_rejected(self):
        bad = copy.deepcopy(self.doc)
        for item in bad["block"]["derivations"]:
            if item["id"] in ("future_threshold", "first_query_threshold"):
                item["note"] = "查询 t 只可见条目 w ≤ (t+1)//M。"
        issues = self.validate(bad)
        self.assertTrue(any("严格可见条件" in i for i in issues), issues)
        self.assertTrue(any("等于阈值" in i for i in issues), issues)

    def test_missing_threshold_derivation_rejected(self):
        bad = copy.deepcopy(self.doc)
        bad["block"]["derivations"] = [
            d for d in bad["block"]["derivations"]
            if d["id"] not in ("future_threshold", "first_query_threshold")]
        issues = self.validate(bad)
        self.assertTrue(any("因果阈值派生量" in i for i in issues), issues)

    def test_missing_cache_notes_rejected(self):
        bad = copy.deepcopy(self.doc)
        bad["block"]["cache_notes"] = []
        self.assertTrue(any("cache_notes" in i for i in self.validate(bad)))

    def test_dangling_tensor_reference_rejected(self):
        bad = copy.deepcopy(self.doc)
        bad["block"]["operators"][0]["inputs"] = ["no_such_tensor"]
        self.assertTrue(any("no_such_tensor" in i for i in self.validate(bad)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
