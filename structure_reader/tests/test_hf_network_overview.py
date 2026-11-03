"""DeepSeek-V4-Flash 整网总览（token→logits）自测。

只测当前源码行为：整网文档事实、四类层组覆盖与层序、外围/终点源码锚点、
契约检查的具名拒绝、以及总览↔四变体详图的真实往返链接。不读旧 outputs、
不依赖既有候选、不导入/运行 HF 模型、不加载权重。

模型输入（根 config.json / transformers 检出）缺失时明确跳过，不伪报通过。

运行方式（工程 Python）：
  cd structure_reader && python3 tests/test_hf_network_overview.py
"""

from __future__ import annotations

import copy
import hashlib
import re
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_KEY = "deepseek_v4_flash_0731_hf"


def _workspace_ready() -> bool:
    return (REPO_ROOT / "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json").is_file() and (
        REPO_ROOT / "hflib/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py"
    ).is_file()


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入（models/ 或 hflib/transformers）；如实跳过")
class TestNetworkOverviewDocument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_network import build_network_overview
        from structure_reader.rules.deepseek_v4_hf import SOURCE_FILES, load_hf_model_values

        cls.build = staticmethod(build_network_overview)
        cls.doc = build_network_overview(REPO_ROOT)
        cls.values = load_hf_model_values(REPO_ROOT)
        cls.input_shas = {
            key: hashlib.sha256((REPO_ROOT / relpath).read_bytes()).hexdigest()
            for key, relpath in SOURCE_FILES.items()
        }

    def test_schema_identity_and_scope(self):
        doc = self.doc
        self.assertEqual(doc["schema"], "hf_network_overview.v1")
        self.assertEqual(doc["model"]["model_key"], MODEL_KEY)
        self.assertEqual(doc["model"]["implementation"], "transformers")
        self.assertNotEqual(doc["model"]["source_revision"], "unknown")
        self.assertEqual(doc["scope"]["entry_class"], "DeepseekV4ForCausalLM")
        self.assertEqual(doc["scope"]["analysis_path"], "eager")
        self.assertEqual(doc["scope"]["layer_count"], self.values.num_hidden_layers)
        self.assertEqual(doc["path"]["id"], "network")
        self.assertEqual([p["id"] for p in doc["path"]["parameters"]][:1], ["B"])

    def test_groups_cover_all_layers_exactly_and_layer_order_is_real(self):
        values = self.values
        groups = self.doc["groups"]
        self.assertEqual([g["id"] for g in groups],
                         ["swa-hash", "csa-hash", "hca-topk", "csa-topk"])
        covered = [i for g in groups for i in g["layer_indices"]]
        self.assertEqual(sorted(covered), list(range(values.num_hidden_layers)))
        self.assertEqual(len(covered), len(set(covered)))
        for group in groups:
            expected = [i for i in range(values.num_hidden_layers)
                        if values.layer_type(i) == group["attention_type"]
                        and values.mlp_layer_type(i) == group["moe_type"]]
            self.assertEqual(group["layer_indices"], expected, group["id"])
            self.assertEqual(group["count"], len(expected))
            self.assertIn(group["representative_layer_index"], expected)
        # 真实层序：逐层顺序与 config 的 layer_types / mlp_layer_types 一致
        order = self.doc["layer_order"]
        self.assertEqual([e["index"] for e in order], list(range(values.num_hidden_layers)))
        for entry in order:
            index = entry["index"]
            self.assertEqual(entry["attention_type"], values.layer_type(index))
            self.assertEqual(entry["moe_type"], values.mlp_layer_type(index))
        # 交错顺序是真实事实（不是“四个连续组”）：0/1 SWA、2 CSA+Hash、3 HCA、4 CSA+TopK
        self.assertEqual([e["group_id"] for e in order[:5]],
                         ["swa-hash", "swa-hash", "csa-hash", "hca-topk", "csa-topk"])

    def test_periphery_and_endpoint_ops_have_real_source_anchors(self):
        ops = {o["id"]: o for o in self.doc["path"]["operators"]}
        expected = {
            "net_embed": "self.embed_tokens",
            "net_mask_build": "create_sliding_window_causal_mask",
            "net_rope_main": "rotary_emb",
            "net_rope_compress": "rotary_emb",
            "net_hc_expand": "hc_mult",
            "net_layer_loop": "for layer in self.layers",
            "net_hc_head_collapse": "sum(dim=2)",
            "net_final_norm": "self.norm",
            "net_logits_slice": "slice_indices",
            "net_lm_head": "self.lm_head",
        }
        locators = {
            "net_embed": "1332-1333", "net_mask_build": "1344-1350",
            "net_layer_loop": "1357-1366", "net_hc_head_collapse": "1003",
            "net_final_norm": "1368", "net_logits_slice": "1510", "net_lm_head": "1511",
        }
        for oid, needle in expected.items():
            self.assertIn(oid, ops)
            blob = " ".join(
                str(r.get("locator", "")) + str(r.get("note", ""))
                for r in ops[oid]["source_refs"]
            )
            self.assertIn(needle, blob, f"{oid} 未对应真实源码")
            source_paths = [str(r.get("path", "")) for r in ops[oid]["source_refs"]
                            if r.get("kind") == "source"]
            self.assertTrue(source_paths, f"{oid} 缺少源码类来源")
            for path in source_paths:
                self.assertTrue(path.endswith(("modeling_deepseek_v4.py",
                                               "masking_utils.py",
                                               "cache_utils.py")),
                                f"{oid} 来源必须是所选源码: {path}")
        for oid, locator in locators.items():
            self.assertIn(locator,
                          " ".join(r.get("locator", "") for r in ops[oid]["source_refs"]),
                          f"{oid} 行列定位漂移")

    def test_boundary_shapes_and_flow_closure(self):
        tensors = {t["id"]: t for t in self.doc["path"]["tensors"]}
        self.assertEqual(tensors["input_ids"]["shape"], ["B", "seqlen"])
        self.assertEqual(tensors["inputs_embeds"]["shape"], ["B", "seqlen", "H"])
        self.assertEqual(tensors["hidden_streams"]["shape"], ["B", "seqlen", "Nhc", "H"])
        self.assertEqual(tensors["hc_head_collapsed"]["shape"], ["B", "seqlen", "H"])
        self.assertEqual(tensors["final_hidden"]["shape"], ["B", "seqlen", "H"])
        self.assertEqual(tensors["lm_input"]["shape"], ["B", "Ssel", "H"])
        self.assertEqual(tensors["logits"]["shape"], ["B", "Ssel", "Vocab"])
        self.assertEqual(tensors["logits"]["kind"], "boundary")
        # 权重不共享：embedding 与 lm_head 是两份独立张量
        self.assertNotEqual(tensors["embed_tokens_weight"]["id"], tensors["lm_head_weight"]["id"])
        self.assertIn("tie_word_embeddings=false", tensors["lm_head_weight"]["note"])
        # 机器核查：数据流闭合（producer 在消费之前、无自环）
        from structure_reader.checks.deepseek_v4_hf_contracts import validate_hf_network_overview
        issues = validate_hf_network_overview(self.doc, self.input_shas)
        self.assertEqual(issues, [], f"当前整网文档不应有问题: {issues[:5]}")

    def test_loop_wiring_and_mhc_not_plain_residual(self):
        ops = {o["id"]: o for o in self.doc["path"]["operators"]}
        loop = ops["net_layer_loop"]
        for tid in ("hidden_streams", "causal_mask", "position_ids", "cos_main",
                    "sin_main", "cos_compress", "sin_compress", "input_ids",
                    "past_key_values"):
            self.assertIn(tid, loop["inputs"], f"层循环未接入真实依赖 {tid}")
        self.assertIn("Nlay", loop["parameter_refs"])
        self.assertIn("streams_after_layers", loop["outputs"])
        self.assertIn("past_key_values_post", loop["outputs"])
        # mHC：入口扩展 + 出口 HyperHead 收拢（不是普通残差模板）
        self.assertIn("Nhc", ops["net_hc_expand"]["parameter_refs"])
        self.assertIn("hc_head_pre", ops["net_hc_head_collapse"]["inputs"])
        self.assertIn("streams_after_layers", ops["net_hc_head_collapse"]["inputs"])
        # 执行顺序：先 HyperHead 收拢、后 Final RMSNorm（源码 :1368）
        order = [o["id"] for o in self.doc["path"]["operators"]]
        self.assertLess(order.index("net_hc_head_collapse"), order.index("net_final_norm"))

    def test_logits_slice_length_clips_at_sequence_length(self):
        from structure_reader.extract.deepseek_v4_network import logits_slice_length

        # 源码 :1510 slice(-Lk, None)：负起点越界裁剪，Lk>T 不扩大长度
        self.assertEqual(logits_slice_length(1, 5), 1)
        self.assertEqual(logits_slice_length(5, 3), 3)
        self.assertEqual(logits_slice_length(5, 5), 5)
        self.assertEqual(logits_slice_length(5, 9), 5)
        self.assertEqual(logits_slice_length(7, 0), 7)
        with self.assertRaises(ValueError):
            logits_slice_length(7, -1)
        with self.assertRaises(ValueError):
            logits_slice_length(7, True)

    def test_document_expresses_clipped_logits_length(self):
        params = {p["id"]: p for p in self.doc["path"]["parameters"]}
        expression = str(params["Ssel"]["expression"])
        note = str(params["Ssel"]["note"])
        self.assertIn("min(seqlen, Lk)", expression)
        self.assertNotIn("S_sel = Lk", expression)
        self.assertIn("seqlen=1、Lk=5", note)
        self.assertIn("S_sel=1", note)
        ops = {o["id"]: o for o in self.doc["path"]["operators"]}
        self.assertIn("min(seqlen, Lk)", str(ops["net_logits_slice"]["note"]))
        self.assertIn("负整数", str(ops["net_logits_slice"]["note"]))

    def test_detail_targets_match_four_groups(self):
        """分层浏览入口内该组的真实路由（同一份结构，无 prefill/decode 双锚点）。"""
        detail = self.doc["detail"]
        self.assertEqual(detail["href"], "structure.html")
        self.assertEqual([g["id"] for g in detail["groups"]],
                         [g["id"] for g in self.doc["groups"]])
        for group in detail["groups"]:
            anchors = group["anchors"]
            gid = group["id"]
            self.assertEqual(anchors["group"], f"/group/{gid}")
            self.assertEqual(anchors["block"], f"/group/{gid}/block")
            self.assertEqual(anchors["operators"], f"/group/{gid}/ops")
            self.assertEqual(anchors["tensors"], f"/group/{gid}/ops")


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入；如实跳过")
class TestNetworkOverviewContractChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_network import build_network_overview
        from structure_reader.rules.deepseek_v4_hf import SOURCE_FILES

        cls.doc = build_network_overview(REPO_ROOT)
        cls.input_shas = {
            key: hashlib.sha256((REPO_ROOT / relpath).read_bytes()).hexdigest()
            for key, relpath in SOURCE_FILES.items()
        }
        from structure_reader.checks.deepseek_v4_hf_contracts import validate_hf_network_overview
        cls.validate = staticmethod(validate_hf_network_overview)

    def test_missing_input_shas_rejected(self):
        issues = self.validate(self.doc, {})
        self.assertTrue(any("input_shas" in i for i in issues))

    def test_dropped_endpoint_op_rejected(self):
        bad = copy.deepcopy(self.doc)
        bad["path"]["operators"] = [o for o in bad["path"]["operators"]
                                    if o["id"] != "net_lm_head"]
        issues = self.validate(bad, self.input_shas)
        self.assertTrue(any("net_lm_head" in i for i in issues), issues)

    def test_broken_mhc_shape_rejected(self):
        bad = copy.deepcopy(self.doc)
        for tensor in bad["path"]["tensors"]:
            if tensor["id"] == "hidden_streams":
                tensor["shape"] = ["B", "T", "H"]
        issues = self.validate(bad, self.input_shas)
        self.assertTrue(any("mHC 流张量" in i for i in issues), issues)

    def test_layer_group_coverage_gap_rejected(self):
        bad = copy.deepcopy(self.doc)
        bad["groups"][0]["layer_indices"] = bad["groups"][0]["layer_indices"][:-1]
        bad["groups"][0]["count"] -= 1
        issues = self.validate(bad, self.input_shas)
        self.assertTrue(any("未恰好覆盖" in i for i in issues), issues)

    def test_loop_missing_mask_dependency_rejected(self):
        bad = copy.deepcopy(self.doc)
        for op in bad["path"]["operators"]:
            if op["id"] == "net_layer_loop":
                op["inputs"] = [t for t in op["inputs"] if t != "causal_mask"]
        issues = self.validate(bad, self.input_shas)
        self.assertTrue(any("causal_mask" in i for i in issues), issues)

    def test_detail_anchor_gap_rejected(self):
        bad = copy.deepcopy(self.doc)
        bad["detail"]["groups"][1]["anchors"]["block"] = ""
        issues = self.validate(bad, self.input_shas)
        self.assertTrue(any("anchors.block" in i for i in issues), issues)

    def test_unclipped_logits_length_claim_rejected(self):
        """回归：把 S_sel 写回 Lk（漏掉 slice 负起点裁剪）必须被门禁拒绝。"""
        bad = copy.deepcopy(self.doc)
        for param in bad["path"]["parameters"]:
            if param["id"] == "Ssel":
                param["expression"] = "S_sel = T（Lk=0）或 S_sel = Lk（Lk>0）"
                param["note"] = "lm_head 只对切片后的 hidden_states 计算，输出 [B,S_sel,Vocab]"
        for op in bad["path"]["operators"]:
            if op["id"] == "net_logits_slice":
                op["note"] = "切片发生在 lm_head 之前；S_sel=T（Lk=0）或 S_sel=Lk（Lk>0）"
        issues = self.validate(bad, self.input_shas)
        self.assertTrue(any("min(seqlen, Lk)" in i for i in issues), issues)


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入；如实跳过")
class TestStructurePageRoundTrip(unittest.TestCase):
    """分层浏览页：整网层 → 四类层组 → Block → 算子 的真实链接与离线约束。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_network import build_network_overview
        from structure_reader.render.hf_structure_page import render_structure_page
        from structure_reader.extract.hf_layer_structures import build_structures_container
        from structure_reader.rules.deepseek_v4_hf import SOURCE_FILES

        cls.structures = build_structures_container(REPO_ROOT)
        cls.network = build_network_overview(REPO_ROOT)
        snapshot = {}
        for relpath in SOURCE_FILES.values():
            path = REPO_ROOT / relpath
            if path.suffix in (".py", ".json"):
                snapshot[relpath] = path.read_text(encoding="utf-8")
        shas = {
            key: hashlib.sha256((REPO_ROOT / relpath).read_bytes()).hexdigest()
            for key, relpath in SOURCE_FILES.items()
        }
        cls.items = [
            {key: v[key] for key in ("id", "label", "attention_type", "moe_type",
                                     "layer_indices", "representative_layer_index",
                                     "structure")}
            for v in cls.structures["variants"]
        ]
        cls.html = render_structure_page(
            cls.network, cls.items, source_snapshot=snapshot,
            input_shas=shas).decode("utf-8")

    def test_network_level_content_and_default_level(self):
        text = self.html
        self.assertIn('id="lv-network"', text)
        self.assertIn('data-lv="network"', text)
        for gid in ("swa-hash", "csa-hash", "hca-topk", "csa-topk"):
            self.assertIn(f'id="lv-group-{gid}"', text)
            self.assertIn(f'href="structure.html#/group/{gid}"', text)
        self.assertEqual(text.count('class="np-layer '), 43)
        # 默认只显示当前层级：CSS 只默认展示 #lv-network，其余靠路由切换
        self.assertIn("#lv-network{display:block}", text.replace("\n", ""))
        self.assertIn(".lv{display:none", text.replace("\n", ""))
        for key in ("modeling_deepseek_v4.py", "configuration_deepseek_v4.py", "config.json"):
            self.assertIn(key, text)
        self.assertIn("tie_word_embeddings=false", text)
        self.assertNotIn("<script src", text)
        self.assertNotRegex(text, r'<link[^>]+href="https?://')
        self.assertNotRegex(text, r'<img[^>]+src="https?://')

    def test_internal_anchors_resolve_and_are_unique(self):
        from html.parser import HTMLParser

        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []
                self.hrefs = []

            def handle_starttag(self, tag, attrs):
                a = dict(attrs)
                if "id" in a:
                    self.ids.append(a["id"])
                href = a.get("href", "")
                if href.startswith("#"):
                    self.hrefs.append(href[1:])
                elif ".html#/" in href:
                    self.hrefs.append(href.split("#", 1)[1])

        parser = Links()
        parser.feed(self.html)
        self.assertEqual(len(parser.ids), len(set(parser.ids)), "页面 id 必须唯一")
        idset = set(parser.ids)
        for target in set(parser.hrefs):
            if target.startswith("/"):
                continue  # 路由：由页面脚本解析，单独校验
            self.assertIn(target, idset, f"内部链接目标缺失: {target}")
        for target in re.findall(r"locateInDiagram\('([^']+)'\)", self.html):
            self.assertIn(f'id="{target}"', self.html, f"框图定位目标缺失: {target}")

    def test_cross_level_routes_resolve(self):
        text = self.html
        routes = {h for h in re.findall(r'href="structure\.html#(/[^"]+)"', text)}
        self.assertIn("/network", routes)
        for group in self.network["groups"]:
            gid = group["id"]
            for route in (f"/group/{gid}", f"/group/{gid}/block", f"/group/{gid}/ops"):
                self.assertIn(route, routes)
        # 每个路由都能落到真实 section；面包屑提供返回上级的普通链接
        for gid in ("swa-hash", "csa-hash", "hca-topk", "csa-topk"):
            self.assertIn(f'id="lv-group-{gid}"', text)
            self.assertIn(f'id="lv-block-{gid}"', text)
            self.assertIn(f'id="lv-ops-{gid}"', text)
        self.assertIn("↑ 返回 Block", text)
        self.assertIn("↑ 返回整网", text)

    def test_svg_renders_real_source_names_and_loop_node(self):
        svg = re.search(r"<svg.*?</svg>", self.html, re.S)
        self.assertIsNotNone(svg)
        body = svg.group(0)
        for needle in ("self.embed_tokens(", "create_sliding_wind",
                       "for layer in self.layers", "self.input_norm(x.f",
                       "self.norm(self.hc_h", "self.lm_head(hidden"):
            self.assertIn(needle, body, f"整网 SVG 未显示真实源码名: {needle}")
        self.assertIn("Loop(×43)", body)


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入；如实跳过")
class TestNetworkMissingSourceRefused(unittest.TestCase):
    def test_missing_config_refused(self):
        from structure_reader.rules.deepseek_v4_hf import HfConfigError

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(HfConfigError):
                from structure_reader.extract.deepseek_v4_network import build_network_overview
                build_network_overview(Path(td))

    def test_unknown_model_key_refused_in_build(self):
        from structure_reader.build_hf_decoder import HfBuildError, run_hf_build

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(HfBuildError) as ctx:
                run_hf_build(REPO_ROOT, Path(td), model_key="not_a_model", all_variants=True)
            self.assertIn("未知模型键", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
