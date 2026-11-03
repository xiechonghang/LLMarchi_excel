"""T14 五模型 Archify 分层图测试：模型声明、分段、来源、形状与跨模型隔离。

运行：cd structure_reader && python3 tests/test_archify_layers_models.py

不实例化模型、不读权重；使用既有的已验收候选文档作为同源事实（受保护历史产物
只读），只构建图与 candidate，不跑 Archify finalize。
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from structure_reader.render import archify_layers as layers  # noqa: E402
from structure_reader.render.archify_profiles import (  # noqa: E402
    MODEL_GRAPH_PROFILES,
    graph_profile_for,
)
from structure_reader.render.hf_model_profile import profile_for  # noqa: E402

# T11 已验收的当前候选（受保护基线；T14 不覆盖这些目录）。
CANDIDATES = {
    "glm_5_3_hf": "1ff1a8282b8b8675352231ca4bc499db9e20ce371362d634a8d12427a0644029",
    "glm_5_3_flash_hf": "e0a72bc340d038c20926ceae09ed648bfad69bf20eed512eecd475db1b0a0cc8",
    "qwen_3_8_flash_next_hf": "7b5bee0e46a04d56bffe1a79dbf1d0d08bdb47271d35b0204b70feafa8e8c17b",
    "hidream_o1_image_hf": "b4cc406ace15beb62637cad18aa2b6f530e98d129cd969b32400ea23831d34de",
}
FORBIDDEN_ALIAS = re.compile(
    r"(^|[^A-Za-z0-9_])(B|S|H|Es|Rk|Rq|Nh|Nkv|Dq|Dv|Dn|Hc|Ttok)([^A-Za-z0-9_]|$)"
)


def _documents(model_key: str):
    base = REPO_ROOT / "structure_reader" / "outputs" / model_key / CANDIDATES[model_key]
    network = json.loads((base / "data/network_overview.json").read_text(encoding="utf-8"))
    shipped = json.loads((base / "data/layer_structures.json").read_text(encoding="utf-8"))
    return base, network, shipped


def _build_graphs(model_key: str):
    base, network, shipped = _documents(model_key)
    profile = graph_profile_for(model_key)
    page = profile_for(model_key)
    graphs = {"network": layers.build_network_graph_for(network, profile)}
    for item in shipped["variants"]:
        gid = item["id"]
        block = item["structure"]["block"]
        group = next(g for g in network["groups"] if g["id"] == gid)
        modules = page.module_groups(block)
        graphs[f"group-{gid}"] = layers.build_group_graph_for(
            network, group, block, modules=modules, profile=profile)
        graphs[f"decoder-{gid}"] = layers.build_decoder_graph_for(
            block, modules=modules, group=group, profile=profile)
        for spec in profile.segments(block, group, modules):
            for form in (("raw", "normalized") if spec.pair_raw else ("normalized",)):
                graphs[f"{spec.level}-{gid}-{form}"] = layers.build_segment_graph(
                    block, group=group, modules=modules, spec=spec,
                    profile=profile, form=form)
    return base, profile, page, network, shipped, graphs


class TestProfiles(unittest.TestCase):
    def test_all_five_models_declared(self):
        self.assertEqual(
            set(MODEL_GRAPH_PROFILES),
            {
                "deepseek_v4_flash_0731_hf", "glm_5_3_hf", "glm_5_3_flash_hf",
                "qwen_3_8_flash_next_hf", "hidream_o1_image_hf",
            },
        )
        for key, profile in MODEL_GRAPH_PROFILES.items():
            self.assertEqual(profile.model_key, key)
            self.assertTrue(profile.levels)
            self.assertIn("network", profile.levels)

    def test_unknown_model_is_named_rejection(self):
        with self.assertRaises(ValueError) as context:
            graph_profile_for("nope")
        self.assertIn("没有 Archify 图声明", str(context.exception))

    def test_hidream_does_not_claim_archify_git_evidence(self):
        self.assertFalse(graph_profile_for("hidream_o1_image_hf").repository_evidence)
        for key in MODEL_GRAPH_PROFILES:
            if key != "hidream_o1_image_hf":
                self.assertTrue(graph_profile_for(key).repository_evidence)

    def test_requests_use_declared_levels_and_raw_pair(self):
        for key in CANDIDATES:
            _, _, shipped = _documents(key)
            profile = graph_profile_for(key)
            page = profile_for(key)
            requests = layers.segment_requests(
                key, shipped["variants"], page.module_groups)
            levels = {request.level for request in requests}
            self.assertLessEqual(levels, set(profile.levels), key)
            self.assertIn("network", levels)
            raw = [request for request in requests if request.form == "raw"]
            for request in raw:
                self.assertEqual(request.level, "attention", key)
            # 每个请求的 (level, variant, form) 唯一。
            triples = {(request.level, request.variant, request.form) for request in requests}
            self.assertEqual(len(triples), len(requests), key)


class TestGraphs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built = {key: _build_graphs(key) for key in CANDIDATES}

    def test_graphs_validate_shapes_and_connectivity(self):
        for key, (base, profile, page, network, shipped, graphs) in self.built.items():
            for name, graph in graphs.items():
                with self.subTest(model=key, graph=name):
                    layers.validate_graph_shapes(graph)
                    positions = layers.layout_level(graph)
                    self.assertEqual(set(positions), {node.node_id for node in graph.nodes})
                    candidate = layers.build_level_candidate(
                        graph,
                        output_relpath=f"viewer/archify/{name}.html",
                        repository=(
                            profile.repository.as_finalize()
                            if profile.repository_evidence else None
                        ),
                        tool=None,
                        source_evidence=profile.repository_evidence,
                    )
                    ids = [component["id"] for component in candidate["components"]]
                    self.assertEqual(len(ids), len(set(ids)))
                    for connection in candidate["connections"]:
                        self.assertIn(connection["from"], ids)
                        self.assertIn(connection["to"], ids)

    def test_lane_connections_keep_label_and_shape(self):
        for key, (base, profile, page, network, shipped, graphs) in self.built.items():
            for slug, graph in graphs.items():
                candidate = layers.build_level_candidate(
                    graph,
                    output_relpath=f"viewer/archify/{slug}.html",
                    repository=None, tool=None, source_evidence=False,
                )
                for connection in candidate["connections"]:
                    if "via" not in connection or connection.get("variant") != "default":
                        continue
                    with self.subTest(model=key, graph=slug,
                                      connection=connection["id"]):
                        self.assertTrue(connection.get("label"),
                                        connection["id"])

    def test_group_loop_uses_real_group_layer_count(self):
        for key, (_base, _profile, _page, network, shipped, graphs) in self.built.items():
            for item in shipped["variants"]:
                graph = graphs[f"group-{item['id']}"]
                loop = next(node for node in graph.nodes if node.node_id == "g-loop")
                with self.subTest(model=key, group=item["id"]):
                    self.assertTrue(loop.label.startswith(
                        f"{len(item['layer_indices'])} 层循环"), loop.label)

    def test_cards_use_full_names(self):
        graphs = self.built["glm_5_3_hf"][5]
        candidate = layers.build_level_candidate(
            graphs["attention-full-moe-normalized"],
            output_relpath="x.html", repository=None, tool=None,
            source_evidence=False,
        )
        cards = {card["title"]: card for card in candidate["cards"]}
        self.assertIn("参数（全称）", cards)
        for item in cards["参数（全称）"]["items"]:
            self.assertFalse(
                re.match(r"^(B|S|H|Nh|Nkv|Dh|Dq|Dk|Dv|Rk|Rq|Es|Hc)[：:]", item),
                item,
            )
        self.assertIn("能力边界", " ".join(
            cards["化简能力与规则（应用 0 / 拒绝 7，原结构保留）"]["items"]))

    def test_index_build_command_includes_archify_flag(self):
        index = REPO_ROOT / "structure_reader" / "docs" / "model_delivery_index.md"
        text = index.read_text(encoding="utf-8")
        self.assertIn("--archify-layers", text)
        self.assertNotIn("build-hf --model <model_key> --all-variants --json", text)

    def test_data_edges_have_full_names_and_no_old_aliases(self):
        for key, (_base, _profile, _page, _network, _shipped, graphs) in self.built.items():
            for name, graph in graphs.items():
                for edge in graph.edges:
                    if edge.kind != "data":
                        continue
                    with self.subTest(model=key, graph=name, tensor=edge.tensor_id):
                        self.assertTrue(edge.shape, f"{name}: {edge.tensor_id} 缺 shape")
                        self.assertIsNone(
                            FORBIDDEN_ALIAS.search(edge.shape),
                            f"{name}: {edge.shape} 泄漏旧别名",
                        )

    def test_sources_are_real_and_model_scoped(self):
        for key, (base, profile, _page, _network, _shipped, graphs) in self.built.items():
            root = REPO_ROOT / profile.repository.root_rel
            for name, graph in graphs.items():
                for node in graph.nodes:
                    for source in node.sources:
                        with self.subTest(model=key, graph=name, node=node.node_id):
                            self.assertFalse(source["path"].startswith("/"))
                            self.assertTrue((root / source["path"]).is_file(),
                                            f"{root / source['path']} 不存在")

    def test_normalization_audit_is_traceable(self):
        for key, (_base, _profile, _page, _network, _shipped, graphs) in self.built.items():
            for name, graph in graphs.items():
                if not name.startswith("attention-") or not name.endswith("-normalized"):
                    continue
                audit = [item for item in graph.expansions if item.get("rule")]
                self.assertEqual(len(audit), 7, f"{key}: {name}")
                for item in audit:
                    self.assertEqual(item["status"], "rejected")
                    self.assertTrue(item["reason"])
                    self.assertIsInstance(item["candidates"], list)

    def test_cross_model_nodes_do_not_leak(self):
        glm_graphs = self.built["glm_5_3_hf"][5]
        qwen_graphs = self.built["qwen_3_8_flash_next_hf"][5]
        glm_ops = {
            op for graph in glm_graphs.values()
            for node in graph.nodes for op in node.op_ids
        }
        qwen_ops = {
            op for graph in qwen_graphs.values()
            for node in graph.nodes for op in node.op_ids
        }
        self.assertTrue(glm_ops)
        self.assertTrue(qwen_ops)
        # 通用 g_moe_*/net_* 名字相同但事实来自各自模型文档；模型特有算子不得互串。
        self.assertTrue(all(op.startswith(("g_", "net_")) for op in glm_ops | qwen_ops))
        # GLM 的 indexer 模块不会出现在 Qwen 的图里，反之亦然。
        self.assertTrue(any("g_ix_" in op for op in glm_ops))
        self.assertFalse(any("g_ix_" in op for op in qwen_ops))
        self.assertTrue(any(op.startswith("g_qsa_") or op.startswith("g_gdn_") for op in qwen_ops))

    def test_shared_state_consumers_keep_each_input_edge(self):
        cases = {
            "glm_5_3_flash_hf": {
                "g_kda_conv_state_pre": {"g_kda_conv_state_update", "g_kda_conv1d"},
                "g_kda_state_pre": {"g_kda_scan", "g_kda_state_write"},
            },
            "qwen_3_8_flash_next_hf": {
                "g_gdn_conv_state_pre": {"g_gdn_conv_state_update", "g_gdn_conv"},
                "g_gdn_state_pre": {"g_gdn_scan", "g_gdn_state_write"},
            },
        }
        for model_key, states in cases.items():
            graphs = self.built[model_key][5]
            for slug, graph in graphs.items():
                if not (slug.startswith("attention-") or slug.startswith("kv-")):
                    continue
                for tensor_id, expected_ops in states.items():
                    edges = [
                        edge for edge in graph.edges
                        if edge.tensor_id == tensor_id and edge.kind == "state"
                        and edge.src.startswith("o-cache-")
                    ]
                    if not edges:
                        continue
                    destinations = {
                        edge.dst.removeprefix("o-") for edge in edges
                    }
                    with self.subTest(model=model_key, graph=slug, state=tensor_id):
                        self.assertEqual(destinations, expected_ops)

    def test_ple_state_consumers_keep_each_input_edge(self):
        graphs = self.built["qwen_3_8_flash_next_hf"][5]
        graph = graphs["ple-gdn-ple-moe-normalized"]
        expected = {
            "g_ple_context_pre": {"g_ple_context_update", "g_ple_history"},
            "g_ple_conv_state_pre": {"g_ple_conv_state_update", "g_ple_conv"},
        }
        for tensor_id, ops in expected.items():
            edges = [
                edge for edge in graph.edges
                if edge.tensor_id == tensor_id and edge.src.startswith("o-cache-")
            ]
            self.assertEqual({edge.dst.removeprefix("o-") for edge in edges}, ops)

    def test_auxiliary_inputs_are_local_branches(self):
        graphs = self.built["glm_5_3_hf"][5]
        graph = graphs["attention-full-moe-normalized"]
        aux_nodes = [
            node for node in graph.nodes if node.node_id.startswith("o-aux-in-")
        ]
        self.assertTrue(aux_nodes)
        for tensor_id in ("g_cos", "g_sin", "g_attention_mask"):
            self.assertTrue(
                any(edge.tensor_id == tensor_id and edge.kind == "aux"
                    for edge in graph.edges),
                tensor_id,
            )
            self.assertFalse(
                any(edge.tensor_id == tensor_id and edge.src == "o-in"
                    for edge in graph.edges),
                tensor_id,
            )
        for node in aux_nodes:
            self.assertEqual(node.role, "aux")
            self.assertTrue(node.anchor and node.anchor.startswith("o-"))
        qwen = self.built["qwen_3_8_flash_next_hf"][5]["attention-qsa-moe-normalized"]
        for tensor_id in ("g_position_cos", "g_position_sin"):
            self.assertTrue(
                any(edge.tensor_id == tensor_id and edge.kind == "aux"
                    for edge in qwen.edges),
                tensor_id,
            )

    def test_display_names_have_no_known_aliases(self):
        aliases = (
            "Stot", "Hq", "Hkv", "Sar", "Fi", "Pin", "Nhv", "Dhv",
            "Nvis", "Svis", "Sgen", "Spix", "Ncnd", "Nar", "Ngt", "Hv",
            "Ptxt", "Himg", "Wimg", "Kref", "Sref", "Ppix", "B", "S",
            "H", "Nh", "Nkv", "Dh", "Es",
        )
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])(" + "|".join(aliases) + r")(?![A-Za-z0-9_])")
        for model_key, (base, profile, page, network, shipped, graphs) in self.built.items():
            for slug, graph in graphs.items():
                candidate = layers.build_level_candidate(
                    graph,
                    output_relpath=f"viewer/archify/{slug}.html",
                    repository=None, tool=None, source_evidence=False,
                )
                for component in candidate["components"]:
                    for field in ("label", "sublabel", "tag"):
                        text = str(component.get(field) or "")
                        with self.subTest(model=model_key, graph=slug,
                                          component=component["id"], field=field):
                            self.assertIsNone(pattern.search(text), text)
                for relation in graph.relations():
                    for field in ("tensor_label", "shape", "label"):
                        text = str(relation.get(field) or "")
                        with self.subTest(model=model_key, graph=slug,
                                          tensor=relation.get("tensor_id"), field=field):
                            self.assertIsNone(pattern.search(text), text)

    def test_K_key_labels_are_not_expert_counts(self):
        from structure_reader.render.display_names import expand_display_text

        for text in ("Q K 投影", "K 交错 RoPE", "K 归一", "K/V 缓存"):
            self.assertEqual(expand_display_text(text), text)
        self.assertEqual(
            expand_display_text("TopK 选 K 个专家"),
            "TopK 选 num_experts_per_tok 个专家",
        )

    def test_qwen_module_titles_use_full_names(self):
        _base, _profile, _page, _network, _shipped, graphs = self.built["qwen_3_8_flash_next_hf"]
        graph = graphs["decoder-qsa-moe"]
        labels = [node.label for node in graph.nodes]
        joined = " ".join(labels)
        self.assertNotIn("[4,H]", joined)
        self.assertIn("hc_count", joined)

    def test_hidream_network_keeps_single_forward_boundary(self):
        _base, _profile, _page, _network, _shipped, graphs = self.built["hidream_o1_image_hf"]
        network = graphs["network"]
        self.assertTrue(any("x_pred" in node.label for node in network.nodes))
        self.assertFalse(any(node.role == "state" for node in network.nodes))
        self.assertTrue(any("net_layer_loop" in node.op_ids for node in network.nodes))
        text = " ".join(node.label for node in network.nodes)
        self.assertIn("x_pred", text)
        self.assertNotIn("lm_head", text)


class TestPageAndRejection(unittest.TestCase):
    def test_block_page_renders_model_specific_subviews(self):
        from structure_reader.render.hf_structure_page import render_structure_page

        base, network, shipped = _documents("qwen_3_8_flash_next_hf")
        entry = {"slug": "x", "level": "ffn", "form": "normalized",
                 "title": "ffn", "html": "viewer/archify/x.html",
                 "candidate": "data/archify/x.json",
                 "relations": "data/archify/edges/x-relations.json",
                 "nodes": 1, "connections": 0, "relations_count": 0, "links": []}
        gid = shipped["variants"][0]["id"]
        panel = {"network": None, "groups": {gid: {
            "group": entry, "decoder": entry,
            "subviews": [{"label": "N-gram PLE 子图（模型声明）", "entries": [entry]}],
            "block_note": "模型声明的模块级说明。",
        }}}
        html = render_structure_page(
            network, shipped["variants"], archify=panel,
            profile=profile_for("qwen_3_8_flash_next_hf")).decode()
        self.assertIn("N-gram PLE 子图（模型声明）", html)
        self.assertIn("模型声明的模块级说明。", html)

    def test_wrapper_links_point_to_real_candidate_data(self):
        page = layers._wrapper_page(
            slug="network", title="t", subtitle="", level="network",
            variant="network", form="normalized", nodes=1, relations_count=0,
            back_href="../structure.html#/network",
            sources="data/archify/sources/network-sources.json",
        ).decode()
        self.assertIn('href="../../data/archify/network.json"', page)
        self.assertIn('href="../../data/archify/sources/network-sources.json"', page)
        self.assertNotIn('href="../data/archify/', page)

    def test_unknown_level_request_is_rejected(self):
        with self.assertRaises(layers.ArchifyLayersError) as context:
            layers.generate_archify_layers(
                REPO_ROOT,
                project_root=REPO_ROOT / "structure_reader",
                network={}, structures=[],
                requests=[layers.LevelRequest("bogus", "g1")],
                repo_root=REPO_ROOT / "hflib" / "transformers",
                repository={},
                model_key="glm_5_3_hf",
            )
        self.assertIn("未知层级", str(context.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
