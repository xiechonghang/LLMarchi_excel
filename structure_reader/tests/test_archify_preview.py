"""Archify 预览生成测试：schema/来源/单向下行/确定性/可移植路径/无个人路径。

运行：cd structure_reader && python3 tests/test_archify_preview.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

REPOSITORY = {
    "url": "https://github.com/huggingface/transformers.git",
    "revision": "536ecc007387a50e77603bb5d92100e9b07514cc",
    "link_mode": "local-only",
}


def _ready() -> bool:
    return (
        REPO_ROOT / "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json"
    ).is_file() and (
        REPO_ROOT / "hflib/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py"
    ).is_file()


@unittest.skipUnless(_ready(), "缺少只读 HF 输入；如实跳过")
class TestArchifyPreview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_semantic import build_semantic_bundle
        from structure_reader.render.archify_preview import (
            build_candidate,
            build_view_model,
            discover_archify,
            tool_identity,
        )
        from structure_reader.semantic.rules import normalize

        cls.build_view_model = staticmethod(build_view_model)
        cls.build_candidate = staticmethod(build_candidate)
        cls.tool = tool_identity(discover_archify(REPO_ROOT))
        cls.raw = build_semantic_bundle(REPO_ROOT, 4)
        cls.result = normalize(
            cls.raw, facts=[tuple(fact) for fact in cls.raw.metadata.get("shape_facts", [])]
        )

    def _candidate(self, form: str, view_key: str) -> dict:
        bundle = self.raw if form == "raw" else self.result.normalized
        view = self.build_view_model(bundle, view_key)
        return self.build_candidate(
            bundle,
            view,
            output_relpath=f"docs/tasks/model-structure-2026-10-28/T12-decoder-verify/previews/test-{view_key}-{form}.html",
            form=form,
            normalization_report=self.result.to_json(),
            tool=self.tool,
            repository=REPOSITORY,
        )

    def test_discovery_prefers_env(self):
        from structure_reader.render.archify_preview import discover_archify

        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "archify.mjs"
            fake.write_text("// fake cli\n", encoding="utf-8")
            original = None
            import os

            original = os.environ.get("ARCHIFY_CLI")
            os.environ["ARCHIFY_CLI"] = str(fake)
            try:
                tool = discover_archify(REPO_ROOT)
                self.assertEqual(tool.cli, fake)
                self.assertTrue(tool.discovery.startswith("env:"))
            finally:
                if original is None:
                    os.environ.pop("ARCHIFY_CLI", None)
                else:
                    os.environ["ARCHIFY_CLI"] = original

    def test_candidate_schema_and_sources(self):
        candidate = self._candidate("normalized", "attention")
        self.assertEqual(candidate["schema_version"], 1)
        self.assertEqual(candidate["diagram_type"], "architecture")
        meta = candidate["meta"]
        self.assertTrue(meta["title"])
        self.assertEqual(meta["locale"], "zh-CN")
        self.assertEqual(meta["quality_profile"], "standard")
        self.assertFalse(meta["output"].startswith("/"))
        self.assertTrue(meta["output"].endswith(".html"))
        self.assertEqual(meta["repository"]["revision"], REPOSITORY["revision"])
        ids = [component["id"] for component in candidate["components"]]
        self.assertEqual(len(ids), len(set(ids)))
        for component in candidate["components"]:
            self.assertIn(component["type"], ("backend", "external"))
            self.assertTrue(component["label"])
            self.assertLessEqual(len(component.get("sources", [])), 3)

    def test_main_chain_single_direction(self):
        for view_key in ("kv", "attention"):
            candidate = self._candidate("normalized", view_key)
            components = {component["id"]: component for component in candidate["components"]}
            for connection in candidate["connections"]:
                self.assertNotEqual(connection["from"], connection["to"])
                self.assertIn(connection["from"], components)
                self.assertIn(connection["to"], components)
                src = components[connection["from"]]
                dst = components[connection["to"]]
                if connection.get("fromSide") == "bottom":
                    self.assertEqual(connection.get("toSide"), "top")
                    self.assertGreater(dst["pos"][1], src["pos"][1] + src["size"][1] - 1)
            # 主链（纵向下行）节点按 y 单调递增
            vertical = [
                (connection["from"], connection["to"])
                for connection in candidate["connections"]
                if connection.get("fromSide") == "bottom"
            ]
            for source_id, target_id in vertical:
                self.assertLess(components[source_id]["pos"][1], components[target_id]["pos"][1])

    def test_no_machine_personal_paths(self):
        for view_key in ("kv", "attention"):
            for form in ("raw", "normalized"):
                text = json.dumps(self._candidate(form, view_key), ensure_ascii=False)
                self.assertNotIn("/Users/", text)
                self.assertNotIn("xiechonghang", text)

    def test_full_name_labels_and_no_alias_tokens(self):
        candidate = self._candidate("normalized", "attention")
        text = json.dumps(
            {
                "labels": [component["label"] for component in candidate["components"]],
                "sublabels": [component.get("sublabel", "") for component in candidate["components"]],
                "edges": [connection.get("label", "") for connection in candidate["connections"]],
            },
            ensure_ascii=False,
        )
        for full_name in ("batch_size", "seqlen", "head_dim", "num_attention_heads"):
            self.assertIn(full_name, text)
        for alias in ("[B,", " Dh", "Rq", "Rk", "Nkv", "Nh,"):
            self.assertNotIn(alias, text)

    def test_candidate_is_deterministic(self):
        first = json.dumps(self._candidate("normalized", "kv"), sort_keys=True, ensure_ascii=False)
        second = json.dumps(self._candidate("normalized", "kv"), sort_keys=True, ensure_ascii=False)
        self.assertEqual(first, second)

    def test_view_node_counts_raw_vs_normalized(self):
        for view_key in ("kv", "attention"):
            raw = self._candidate("raw", view_key)
            normalized = self._candidate("normalized", view_key)
            if view_key == "kv":
                self.assertEqual(len(raw["components"]), 8)
                self.assertEqual(len(normalized["components"]), 7)
            else:
                self.assertGreaterEqual(len(raw["components"]), 36)
                self.assertLess(len(normalized["components"]), len(raw["components"]))

    def test_rewritten_node_exposes_original_op(self):
        normalized = self._candidate("normalized", "kv")
        tags = [component.get("tag", "") for component in normalized["components"]]
        self.assertTrue(any("原 kv_head_view" in tag for tag in tags), tags)
        labels = [component["label"] for component in normalized["components"]]
        self.assertTrue(any("Reshape" in label for label in labels), labels)

    def test_simplification_card_lists_rules(self):
        candidate = self._candidate("normalized", "kv")
        titles = [card["title"] for card in candidate["cards"]]
        self.assertTrue(any("化简对照" in title for title in titles), titles)
        text = json.dumps(candidate["cards"], ensure_ascii=False)
        for rule_id in ("R1", "R3", "R5", "R7"):
            self.assertIn(rule_id, text)




class TestBrowserClassification(unittest.TestCase):
    def test_environment_failure_only_for_runtime_code(self):
        from structure_reader.render.archify_preview import classify_browser_failure

        receipt = {
            "failedStage": "browser-check",
            "gates": {"validate": "pass", "deliver": "pass", "check": "pass"},
            "diagnostics": [
                {
                    "code": "viewer/browser-check-runtime",
                    "severity": "error",
                    "message": "Chrome DevTools read pipe failed",
                }
            ],
        }
        kind, detail = classify_browser_failure(receipt)
        self.assertEqual(kind, "environment")
        self.assertIn("Chrome", detail)

    def test_geometry_failure_is_real(self):
        from structure_reader.render.archify_preview import classify_browser_failure

        receipt = {
            "failedStage": "browser-check",
            "diagnostics": [
                {"code": "composition/proper-crossing", "severity": "error", "message": "cross"},
                {"code": "viewer/browser-check-runtime", "severity": "error", "message": "chrome"},
            ],
        }
        self.assertEqual(classify_browser_failure(receipt)[0], "real")

    def test_evidence_conflict_and_other_stages_are_real(self):
        from structure_reader.render.archify_preview import classify_browser_failure

        self.assertEqual(
            classify_browser_failure(
                {
                    "failedStage": "browser-check",
                    "diagnostics": [
                        {"code": "viewer/evidence-path-conflict", "severity": "error", "message": "x"}
                    ],
                }
            )[0],
            "real",
        )
        self.assertEqual(
            classify_browser_failure(
                {"failedStage": "validate", "diagnostics": []}
            )[0],
            "real",
        )


class TestPackageClosure(unittest.TestCase):
    def test_closure_digest_is_change_sensitive(self):
        from structure_reader.render.archify_preview import package_closure

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.mjs").write_text("export const a = 1;\n", encoding="utf-8")
            (root / "sub").mkdir()
            (root / "sub" / "b.mjs").write_text("export const b = 2;\n", encoding="utf-8")
            first = package_closure(root)
            second = package_closure(root)
            self.assertEqual(first["digest"], second["digest"])
            self.assertEqual(first["file_count"], 2)
            (root / "sub" / "b.mjs").write_text("export const b = 3;\n", encoding="utf-8")
            changed = package_closure(root)
            self.assertNotEqual(first["digest"], changed["digest"])
            self.assertEqual(changed["files"][1]["path"], "sub/b.mjs")

    def test_real_tool_identity_includes_dependency_closure(self):
        from structure_reader.render.archify_preview import discover_archify, tool_identity

        identity = tool_identity(discover_archify(REPO_ROOT))
        package = identity["package"]
        self.assertGreater(package["file_count"], 100)
        self.assertEqual(len(package["digest"]), 64)
        self.assertNotIn("cli", identity)
        self.assertNotIn("root", package)
        self.assertNotIn("files", package)
        self.assertNotIn(str(REPO_ROOT), json.dumps(identity))

    def test_identity_is_portable_across_home_and_install_path(self):
        import os
        from unittest.mock import patch

        from structure_reader.render.archify_preview import discover_archify, tool_identity

        def make_home(root: Path) -> Path:
            package = root / ".dsh" / "skills" / "archify"
            (package / "bin").mkdir(parents=True)
            (package / "renderers").mkdir()
            (package / "bin" / "archify.mjs").write_text(
                "export const cli = 1;\n", encoding="utf-8")
            (package / "renderers" / "a.mjs").write_text(
                "export const a = 2;\n", encoding="utf-8")
            (package / "package.json").write_text(
                '{"name":"archify","version":"3.2.0"}\n', encoding="utf-8")
            (package / "skill-release.json").write_text(
                '{"version":"3.2.0"}\n', encoding="utf-8")
            return root

        original = os.environ.pop("ARCHIFY_CLI", None)
        try:
            with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
                home_one = make_home(Path(one))
                home_two = make_home(Path(two))
                with patch.object(Path, "home", return_value=home_one):
                    first = tool_identity(discover_archify(REPO_ROOT))
                with patch.object(Path, "home", return_value=home_two):
                    second = tool_identity(discover_archify(REPO_ROOT))
                self.assertEqual(first, second)
                rendered = json.dumps(first)
                self.assertNotIn(str(home_one), rendered)
                self.assertNotIn(str(home_two), rendered)
                self.assertNotIn("bin/archify.mjs", rendered)
        finally:
            if original is not None:
                os.environ["ARCHIFY_CLI"] = original


@unittest.skipUnless(_ready(), "缺少只读 HF 输入；如实跳过")
class TestViewKeepsRealDependencies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_semantic import build_semantic_bundle
        from structure_reader.render.archify_preview import build_view_model
        from structure_reader.semantic.rules import normalize

        cls.raw = build_semantic_bundle(REPO_ROOT, 4)
        cls.result = normalize(
            cls.raw, facts=[tuple(fact) for fact in cls.raw.metadata.get("shape_facts", [])]
        )
        cls.build_view_model = staticmethod(build_view_model)

    def test_every_member_data_edge_has_a_trace_record(self):
        for form, bundle in (("raw", self.raw), ("normalized", self.result.normalized)):
            for view_key in ("kv", "attention"):
                view = self.build_view_model(bundle, view_key)
                spec = bundle.metadata["views"][view_key]
                old_ops = set(spec["old_ops"]) | set(spec.get("extra_old_ops", ()))
                members = {
                    node_id
                    for node_id, meta in bundle.node_meta.items()
                    if any(ref == f"op:{op}" for ref in meta.legacy_refs for op in old_ops)
                    and not node_id.startswith("layer.")
                }
                expected = {
                    edge.stable_id
                    for edge in bundle.graph.edges
                    if edge.kind == "data"
                    and edge.src_node_id in members
                    and edge.dst_node_id in members
                }
                recorded = {
                    record["semantic_edge"]
                    for record in view.edge_records
                    if record["semantic_edge"] != "(boundary)"
                    and record["kind"] == "data"
                    and record["src_node"] in members
                    and record["dst_node"] in members
                }
                self.assertEqual(expected, recorded, f"{form}:{view_key}")

    def test_boundary_input_reaches_every_consumer(self):
        view = self.build_view_model(self.result.normalized, "attention")
        targets = {
            connection.dst
            for connection in view.connections
            if connection.src == view.boundary_input
        }
        self.assertIn("attn.q_a_proj", targets)
        self.assertIn("attn.kv_proj", targets)
        self.assertIn("attn.compressor", targets)
        self.assertIn("attn.indexer", targets)

    def test_kv_proj_has_an_input_and_scale_mask_edge_exists(self):
        view = self.build_view_model(self.result.normalized, "attention")
        self.assertTrue(any(c.dst == "attn.kv_proj" for c in view.connections))
        self.assertTrue(
            any(c.src == "attn.scale" and c.dst == "attn.mask_add" for c in view.connections)
        )
        labels = {
            (c.src, c.dst): c.label for c in view.connections
        }
        self.assertIn("[batch_size, num_attention_heads, seqlen, kv_length + num_compressed]",
                      labels.get(("attn.scale", "attn.mask_add"), ""))

    def test_state_edges_are_traceable(self):
        view = self.build_view_model(self.result.normalized, "attention")
        records = {
            record["semantic_edge"]: record
            for record in view.edge_records
            if record["kind"] == "state"
        }
        self.assertTrue(records)
        for record in records.values():
            self.assertTrue(record["state_transition_ref"])
            self.assertTrue(record["tensor_id"].startswith("tensor:"))

    def test_layer_hc_post_and_comb_dependencies(self):
        bundle = self.raw
        attn_hc = bundle.nodes["layer.attn_hc"]
        attn_residual = bundle.nodes["layer.attn_residual"]
        self.assertEqual(
            {port.name for port in attn_hc.output_ports}, {"collapsed", "post", "comb"}
        )
        self.assertTrue(
            {"post", "comb"}.issubset({port.name for port in attn_residual.input_ports})
        )
        edges = {
            (edge.src_node_id, edge.src_port, edge.dst_node_id, edge.dst_port)
            for edge in bundle.graph.edges
            if edge.kind == "data"
        }
        self.assertIn(("layer.attn_hc", "post", "layer.attn_residual", "post"), edges)
        self.assertIn(("layer.attn_hc", "comb", "layer.attn_residual", "comb"), edges)
        self.assertIn(("layer.ffn_hc", "post", "layer.ffn_residual", "post"), edges)
        self.assertIn(("layer.ffn_hc", "comb", "layer.ffn_residual", "comb"), edges)
        comb_shape = bundle.shape_text("tensor:attn_mhc_comb")
        self.assertIn("hc_mult（源流 j）", comb_shape)
        self.assertIn("hc_mult（目标流 k）", comb_shape)






@unittest.skipUnless(_ready(), "缺少只读 HF 输入；如实跳过")
class TestStateEdgesDownward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_semantic import build_all_layer_group_bundles
        from structure_reader.render.archify_preview import _layout, build_view_model
        from structure_reader.semantic.rules import normalize

        cls.build_view_model = staticmethod(build_view_model)
        cls.layout = staticmethod(_layout)
        bundles = build_all_layer_group_bundles(REPO_ROOT)
        cls.cases = []
        for variant, bundle in bundles.items():
            result = normalize(
                bundle,
                facts=[tuple(fact) for fact in bundle.metadata.get("shape_facts", [])],
            )
            cls.cases.append((variant, "raw", bundle))
            cls.cases.append((variant, "normalized", result.normalized))

    def test_all_state_edges_point_downward_in_layout(self):
        for variant, form, bundle in self.cases:
            for view_key in ("kv", "attention"):
                view = self.build_view_model(bundle, view_key)
                positions = self.layout(view)
                dashed = [item for item in view.connections if item.variant == "dashed"]
                for connection in dashed:
                    source = positions[connection.src]
                    target = positions[connection.dst]
                    self.assertLess(
                        source[1],
                        target[1],
                        f"{variant}:{form}:{view_key}: {connection.src}→{connection.dst}",
                    )
                # 前态在所有消费者上方，后态在所有生产者下方
                for node in view.nodes:
                    if node.role != "state":
                        continue
                    if node.semantic_id.endswith("cache_in"):
                        targets = [c.dst for c in dashed if c.src == node.semantic_id]
                        self.assertTrue(targets)
                        self.assertLess(
                            positions[node.semantic_id][1],
                            min(positions[target][1] for target in targets),
                        )
                    else:
                        sources = [c.src for c in dashed if c.dst == node.semantic_id]
                        self.assertTrue(sources)
                        self.assertGreater(
                            positions[node.semantic_id][1],
                            max(
                                positions[source][1] + positions[source][3]
                                for source in sources
                            ),
                        )

    def test_route_violation_detector(self):
        from structure_reader.render.archify_preview import state_route_violations

        down = '<path class="a-dashed" data-edge-id="e1" data-composition-points="10,100;40,100;40,300;10,300"/>'
        up = '<path class="a-dashed" data-edge-id="e2" data-composition-points="10,300;40,300;40,100;10,100"/>'
        self.assertEqual(state_route_violations(down), [])
        violations = state_route_violations(up)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["edge_id"], "e2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
