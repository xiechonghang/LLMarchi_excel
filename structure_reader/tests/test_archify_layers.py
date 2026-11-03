"""T13 分层 Archify 图测试：层图构建、聚合不丢张量、单向布局、页面入口与模型闸门。

运行：cd structure_reader && python3 tests/test_archify_layers.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from structure_reader.build_hf_decoder import (  # noqa: E402
    _build_archify_layers as layers_build_archify,
)
from structure_reader.render import archify_layers as layers  # noqa: E402
from structure_reader.render.display_names import shape_text  # noqa: E402

REPOSITORY = {
    "url": "https://github.com/huggingface/transformers.git",
    "revision": "536ecc007387a50e77603bb5d92100e9b07514cc",
    "link_mode": "local-only",
}
SOURCE = {
    "kind": "source",
    "path": "hflib/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py",
    "locator": "700-704",
    "note": "fixture",
}


WEIGHT_NAMES = {
    "shared_gate_weight": "shared_experts.gate_proj.weight",
    "shared_up_weight": "shared_experts.up_proj.weight",
    "shared_down_weight": "shared_experts.down_proj.weight",
    "expert_gate_up_weight": "gate_up_proj[expert_idx=e]",
    "expert_down_weight": "down_proj[expert_idx=e]",
}


def _tensor(tid, shape, kind="activation"):
    return {
        "id": tid,
        "kind": kind,
        "name": WEIGHT_NAMES.get(tid, tid),
        "shape": shape,
        "parameter_refs": [],
    }


def _op(oid, module, inputs, outputs, name=None, op="forward", weights=None):
    return {
        "id": oid,
        "module_path": module,
        "name": name or oid,
        "op": op,
        "inputs": inputs,
        "outputs": outputs,
        "weights": list(weights or ()),
        "source_refs": [SOURCE],
        "note": "",
        "parameter_refs": [],
    }


def _network_document() -> dict:
    path = {
        "id": "network",
        "parameters": [{"id": "B", "symbol": "B", "meaning": "batch", "value": None}],
        "tensors": [
            _tensor("input_ids", ["B", "seqlen"], "boundary"),
            _tensor("inputs_embeds", ["B", "seqlen", "H"]),
            _tensor("hidden_streams", ["B", "seqlen", "Nhc", "H"]),
            _tensor("streams_after_layers", ["B", "seqlen", "Nhc", "H"]),
            _tensor("logits", ["B", "seqlen", "Vocab"], "boundary"),
            _tensor("past_key_values", ["layers"], "cache"),
            _tensor("past_key_values_post", ["layers"], "cache"),
            _tensor("attention_mask", ["B", "seqlen"], "boundary"),
        ],
        "operators": [
            _op("net_embed", "Model.embed_tokens", ["input_ids"], ["inputs_embeds"]),
            _op("net_mask_build", "Model.forward", ["inputs_embeds", "attention_mask"],
                ["causal_mask"]),
            _op("net_hc_expand", "Model.forward", ["inputs_embeds"], ["hidden_streams"]),
            _op("net_layer_loop", "Model.layers", ["hidden_streams", "past_key_values"],
                ["streams_after_layers", "past_key_values_post"]),
            _op("net_lm_head", "ForCausalLM.lm_head", ["streams_after_layers"], ["logits"]),
        ],
    }
    return {"schema": "hf_network_overview.v1",
            "scope": {"entry_class": "ForCausalLM", "layer_count": 43, "hidden_size": 4096},
            "groups": [], "path": path, "layer_order": []}


def _block_document() -> dict:
    tensors = [
        _tensor("hidden_streams_in", ["B", "seqlen", "Nhc", "H"], "boundary"),
        _tensor("hidden_streams_out", ["B", "seqlen", "Nhc", "H"], "boundary"),
        _tensor("attn_mhc_collapsed", ["B", "seqlen", "H"]),
        _tensor("attn_out_streams", ["B", "seqlen", "Nhc", "H"]),
        _tensor("attn_mhc_post", ["B", "seqlen", "Nhc"]),
        _tensor("attn_mhc_comb", ["B", "Nhc", "Nhc"]),
        _tensor("ffn_mhc_collapsed", ["B", "seqlen", "H"]),
        _tensor("shared_gate", ["B", "seqlen", "Es"]),
        _tensor("shared_up", ["B", "seqlen", "Es"]),
        _tensor("shared_out", ["B", "seqlen", "H"]),
        _tensor("shared_act", ["B", "seqlen", "Es"]),
        _tensor("mlp_output", ["B", "seqlen", "H"]),
        _tensor("ffn_expanded", ["B", "seqlen", "Nhc", "H"]),
        _tensor("expert_gu_out", ["N_e", "2Es"]),
        _tensor("expert_down_raw", ["N_e", "H"]),
        _tensor("expert_gate_up_weight", ["2Es", "H"], "weight"),
        _tensor("expert_down_weight", ["H", "Es"], "weight"),
        _tensor("ffn_mhc_post", ["B", "seqlen", "Nhc"]),
        _tensor("ffn_mhc_comb", ["B", "Nhc", "Nhc"]),
        _tensor("cache_keys", ["B", "1", "kvcachelen", "Dh"], "cache"),
        _tensor("cache_keys_post", ["B", "1", "min(kvcachelen+seqlen,W-1)", "Dh"], "cache"),
        _tensor("cache_buffer_c_kv", ["B", "W", "Dh"], "cache"),
        _tensor("cache_buffer_c_kv_post", ["B", "W", "Dh"], "cache"),
        _tensor("attn_normed", ["B", "seqlen", "H"]),
        _tensor("attn_block_output", ["B", "seqlen", "H"]),
        _tensor("q_a_norm_weight", ["Rq"], "weight"),
        _tensor("shared_gate_weight", ["Es", "H"], "weight"),
        _tensor("shared_up_weight", ["Es", "H"], "weight"),
        _tensor("shared_down_weight", ["H", "Es"], "weight"),
    ]
    operators = [
        _op("attn_hc_flatten", "Layer.attn_hc", ["hidden_streams_in"], ["attn_mhc_collapsed"]),
        _op("attn_hc_post", "Layer.attn_hc", ["attn_mhc_collapsed"], ["attn_mhc_post"]),
        _op("attn_hc_comb", "Layer.attn_hc", ["attn_mhc_collapsed"], ["attn_mhc_comb"]),
        _op("attn_norm", "Layer.input_layernorm", ["attn_mhc_collapsed", "q_a_norm_weight"],
            ["attn_normed"]),
        _op("o_b_proj", "Attention", ["attn_normed"], ["attn_block_output"]),
        _op("cache_update", "Attention", ["attn_block_output", "cache_keys", "cache_buffer_c_kv"],
            ["cache_keys_post", "cache_buffer_c_kv_post"]),
        _op("attn_residual_expand", "Layer.forward",
            ["attn_mhc_post", "attn_block_output"], ["attn_out_streams"]),
        _op("attn_residual_mix", "Layer.forward",
            ["attn_mhc_comb", "hidden_streams_in", "attn_out_streams"], ["ffn_mhc_collapsed"]),
        _op("shared_gate_up", "DeepseekV4MLP", ["ffn_mhc_collapsed"],
            ["shared_gate", "shared_up"], op="linear+clamp×2",
            weights=["shared_gate_weight", "shared_up_weight"]),
        _op("shared_act_down", "DeepseekV4MLP", ["shared_gate", "shared_up"],
            ["shared_act", "shared_out"], op="silu+mul+linear",
            weights=["shared_down_weight"]),
        _op("moe_expert_gu", "DeepseekV4Experts（专家模板）", ["ffn_mhc_collapsed"],
            ["expert_gu_out"], op="linear", weights=["expert_gate_up_weight"]),
        _op("moe_expert_down", "DeepseekV4Experts（专家模板）", ["expert_gu_out"],
            ["expert_down_raw"], op="linear", weights=["expert_down_weight"]),
        _op("moe_merge", "DeepseekV4SparseMoeBlock", ["shared_out", "expert_down_raw"],
            ["mlp_output"]),
        _op("ffn_residual_expand", "Layer.forward", ["ffn_mhc_post", "mlp_output"],
            ["ffn_expanded"]),
        _op("ffn_residual_mix", "Layer.forward",
            ["ffn_mhc_comb", "attn_out_streams", "ffn_expanded"], ["hidden_streams_out"]),
    ]
    return {
        "schema": "hf_layer_structure.v1",
        "layer_index": 4,
        "tensors": tensors,
        "operators": operators,
        "parameters": [{"id": "B", "symbol": "B", "meaning": "batch"}],
        "cache_notes": ["cache fixture"],
        "coverage_notes": ["coverage fixture"],
        "scope_notes": ["scope fixture"],
        "title": "fixture block",
    }


GROUP = {"id": "g1", "label": "G1", "layer_indices": [0, 1], "count": 2,
         "attention_type": "sliding_attention", "moe_type": "moe",
         "representative_layer_index": 0}


class TestLevelGraphs(unittest.TestCase):
    def test_network_graph_tracks_state_and_aux(self):
        graph = layers.build_network_graph(_network_document())
        kinds = {edge.kind for edge in graph.edges}
        self.assertIn("state", kinds)
        self.assertIn("aux", kinds)
        node_ids = {node.node_id for node in graph.nodes}
        self.assertIn("n-cache-in", node_ids)
        self.assertIn("n-cache-out", node_ids)
        self.assertIn("n-input", node_ids)

    def test_network_uses_full_shape_names(self):
        graph = layers.build_network_graph(_network_document())
        shapes = " ".join(edge.shape for edge in graph.edges)
        self.assertIn("batch_size", shapes)
        self.assertIn("hidden_size", shapes)
        self.assertNotIn("B,seqlen", shapes)

    def test_group_graph_keeps_real_layers_and_template(self):
        block = _block_document()
        modules = [{"id": "mhc_attn", "title": "mHC", "ops": block["operators"]}]
        graph = layers.build_group_graph(_network_document(), GROUP, block, modules=modules)
        group_node = graph.node("g-group")
        self.assertIn("0", group_node.sublabel)
        self.assertIn("1", group_node.sublabel)
        block_node = graph.node("g-block")
        self.assertIn("1 模块", block_node.label)

    def test_decoder_splits_post_and_comb_into_decoder_forward_nodes(self):
        from structure_reader.render.diagram import module_groups

        block = _block_document()
        modules = module_groups(block)
        graph = layers.build_decoder_graph(
            block, modules=modules, group=GROUP, attention_types={}
        )
        node_ids = {node.node_id for node in graph.nodes}
        # post 展开、comb 转置、combᵀ 混合+Add 各自是 Decoder.forward 节点，
        # 不再被压成一个“残差”阶段。
        # 该合成 fixture 没有 combT 算子；真实 DeepSeek block 的 combT
        # 回归在 test_deepseek_reading_page 的 D02 用例中覆盖。
        for node_id in (
            "m-attn_residual_expand",
            "m-attn_residual_mix",
            "m-ffn_residual_expand",
            "m-ffn_residual_mix",
        ):
            self.assertIn(node_id, node_ids)
        # mHC 只向 Norm/post/comb 三个消费点供数，不能直接连到 Attention。
        attn_hc = next(edge for edge in graph.edges
                       if edge.dst == "m-norm_attn#1" and edge.src.startswith("m-mhc_attn"))
        self.assertEqual(attn_hc.label, "collapsed")
        post = next(edge for edge in graph.edges
                    if edge.dst == "m-attn_residual_expand" and edge.src.startswith("m-mhc_attn"))
        self.assertEqual(post.label, "post")
        mix_inputs = {
            edge.tensor_id for edge in graph.edges
            if edge.dst == "m-attn_residual_mix"
        }
        self.assertIn("attn_mhc_comb", mix_inputs)
        self.assertIn("attn_out_streams", mix_inputs)
        self.assertIn("hidden_streams_in", mix_inputs)
        self.assertFalse(any(
            edge.src.startswith("m-mhc_attn") and edge.dst.startswith("m-attn_q")
            for edge in graph.edges
        ))

    def test_decoder_keeps_one_cache_module_state_pair(self):
        from structure_reader.render.diagram import module_groups

        block = _block_document()
        graph = layers.build_decoder_graph(
            block, modules=module_groups(block), group=GROUP, attention_types={}
        )
        state_nodes = [node for node in graph.nodes if node.role == "state"]
        self.assertEqual(len(state_nodes), 2, [node.node_id for node in state_nodes])
        # 只画算子最少的缓存模块（主 KV），缓冲/overlap 等留在 Attention 层。
        self.assertTrue(state_nodes[0].node_id.startswith("d-cache-cache-keys"))
        self.assertTrue(state_nodes[1].node_id.startswith("d-cache-cache-keys-post"))

    def test_moe_and_mhc_graphs_use_operator_granularity(self):
        from structure_reader.render.diagram import module_groups

        block = _block_document()
        modules = module_groups(block)
        moe = layers.build_moe_graph(block, modules=modules, group=GROUP)
        # 共享专家按具名 Linear 展开（gate/up/down），不再保留复合框
        labels = {node.label for node in moe.nodes}
        self.assertIn("gate_proj · Linear", labels)
        self.assertIn("up_proj · Linear", labels)
        self.assertIn("down_proj · Linear", labels)
        self.assertFalse(any(node.node_id == "o-shared_gate_up" for node in moe.nodes))
        mhc = layers.build_mhc_graph(block, modules=modules, group=GROUP)
        op_ids = {node.node_id for node in mhc.nodes}
        self.assertIn("o-attn_hc_post", op_ids)
        self.assertIn("o-attn_hc_comb", op_ids)


class TestLayout(unittest.TestCase):
    def test_strongly_connected_pairs_share_rank(self):
        graph = layers.LevelGraph(key="t", level="decoder", title="t", subtitle="")
        graph.nodes = [
            layers.LevelNode("a", "a", "", "t", "backend", (), "main"),
            layers.LevelNode("b", "b", "", "t", "backend", (), "main"),
        ]
        graph.edges = [
            layers.LevelEdge("a", "b", "t1", "t1", "", "data"),
            layers.LevelEdge("b", "a", "t2", "t2", "", "data"),
        ]
        ranks = layers._ranks(graph)
        self.assertEqual(ranks["a"], ranks["b"])
        positions = layers.layout_level(graph)
        self.assertEqual(positions["a"][1], positions["b"][1])

    def test_data_edges_never_go_backwards(self):
        graph = layers.build_network_graph(_network_document())
        positions = layers.layout_level(graph)
        rank = layers._ranks(graph)
        for edge in graph.edges:
            if edge.kind != "data":
                continue
            # 辅助预计算在侧栏（role=branch），不参与主链秩；它们的边由
            # 位置方向测试单独覆盖（test_side_inputs_flow_downward）。
            if edge.src not in rank or edge.dst not in rank:
                continue
            self.assertGreaterEqual(rank[edge.dst], rank[edge.src])


class TestCandidate(unittest.TestCase):
    def test_candidate_sources_are_repo_relative_and_capped(self):
        graph = layers.build_network_graph(_network_document())
        candidate = layers.build_level_candidate(
            graph,
            output_relpath="viewer/archify/network.html",
            repository=REPOSITORY,
            tool={"version": "test", "cli_sha256": "0" * 64, "node": "v0"},
        )
        self.assertEqual(candidate["diagram_type"], "architecture")
        self.assertNotIn("viewBox", candidate["meta"])
        for component in candidate["components"]:
            self.assertLessEqual(len(component.get("sources", [])), 3)
            for source in component.get("sources", []):
                self.assertFalse(source["path"].startswith("hflib/"))
        cards = {card["title"] for card in candidate["cards"]}
        self.assertIn("节点全称（主图截断文字的完整版本）", cards)

    def test_label_fit_respects_pixel_budget(self):
        self.assertLessEqual(layers._text_width(layers._fit("x" * 200, 120, 13), 13), 120)

    def test_level_requests_cover_all_levels_and_raw_pair(self):
        requests = []
        from structure_reader.build_hf_decoder import _archify_level_requests

        requests = _archify_level_requests(["swa-hash", "csa-topk"])
        slugs = {(request.level, request.variant, request.form) for request in requests}
        for level in ("network", "embedding", "head", "group", "decoder", "kv",
                      "attention", "moe", "mhc"):
            self.assertTrue(any(item[0] == level for item in slugs), level)
        self.assertIn(("kv", "csa-topk", "raw"), slugs)
        self.assertIn(("attention", "csa-topk", "raw"), slugs)
        # 整网 + 两个简单端点（D01）+ 每组 6 层 + 2 张 raw 对照
        # D03：每组 cache-main/q/o；CSA 另有缓存压缩器/压缩器/缓存Indexer/Indexer。
        # 两个合成层组：SWA 9 + CSA 13 + 两个 raw 对照 + 3 端点。
        # D04：每组 +4 张 MoE 关系/路由/专家/共享/汇合层；CSA 另有既有 D03 扩展。
        self.assertEqual(len(requests), 3 + 13 + 17 + 2)
        self.assertIn(("embedding", "network", "normalized"), slugs)
        self.assertIn(("head", "network", "normalized"), slugs)


class TestPagePanel(unittest.TestCase):
    """用既有真实候选文档渲染分层页（含/不含 Archify 入口），不重建产物。"""

    CANDIDATE = (
        REPO_ROOT
        / "structure_reader/outputs/deepseek_v4_flash_0731_hf"
        / "c4a2f98acd52a8f40b08481a84771c58f276957c6bbd9acf2da3900d24e9c4a5"
    )

    def _documents(self):
        if not (self.CANDIDATE / "data/network_overview.json").is_file():
            self.skipTest("缺少既有候选文档；如实跳过")
        network = json.loads(
            (self.CANDIDATE / "data/network_overview.json").read_text(encoding="utf-8")
        )
        shipped = json.loads(
            (self.CANDIDATE / "data/layer_structures.json").read_text(encoding="utf-8")
        )
        return network, shipped["variants"]

    def test_panel_absent_by_default(self):
        from structure_reader.render.hf_structure_page import render_structure_page

        network, structures = self._documents()
        html = render_structure_page(network, structures).decode("utf-8")
        self.assertNotIn('class="ac-panel"', html)

    def test_panel_renders_links_and_back_context(self):
        from structure_reader.render.hf_structure_page import render_structure_page

        network, structures = self._documents()
        entry = {
            "slug": "decoder-g1", "level": "decoder", "form": "normalized",
            "title": "decoder", "html": "viewer/archify/decoder-g1.html",
            "candidate": "data/archify/decoder-g1.json",
            "relations": "data/archify/edges/decoder-g1-relations.json",
            "nodes": 3, "connections": 2, "relations_count": 2,
            "gates": {"validate": "pass", "deliver": "pass", "check": "pass"},
            "browser_check_available": True,
            "links": [{"from": "a", "to": "b", "tensor": "t",
                       "shape": "batch_size,seqlen", "kind": "data", "label": "x"}],
        }
        gid = str(structures[0]["id"])
        panel = {"network": entry, "groups": {gid: {
            "group": entry, "decoder": entry, "kv": [entry], "attention": [entry],
            "moe": entry, "mhc": entry}}}
        html = render_structure_page(network, structures, archify=panel).decode("utf-8")
        self.assertIn('class="ac-panel"', html)
        self.assertIn("archify/decoder-g1.html", html)
        self.assertIn("../data/archify/edges/decoder-g1-relations.json", html)
        self.assertIn("↑ 返回层组", html)
        self.assertIn("↑ 返回整网", html)
        self.assertIn("batch_size,seqlen", html)
        # 子图入口走包装页（可返回本层）；原始图另开新标签。
        self.assertIn("打开分层子图（可返回本层）", html)
        self.assertEqual(html.count('target="_blank"'), html.count("原始图新标签"))


class TestRoundTwoStandards(unittest.TestCase):
    """Codex 第一轮标准轴 3 项：首次构建、跨环境稳定交付、显式开关不静默。"""

    def test_fresh_work_directory_is_created(self):
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory(prefix="t13-fresh-") as directory:
            project = Path(directory)
            from structure_reader.render.archify_preview import ArchifyToolError

            with patch(
                "structure_reader.render.archify_layers.discover_archify",
                side_effect=ArchifyToolError("no tool in fixture"),
            ):
                # 无可用工具时也不能因为 .work 父目录缺失而 FileNotFoundError
                with self.assertRaises(ArchifyToolError):
                    layers.generate_archify_layers(
                        REPO_ROOT,
                        project_root=project,
                        network={},
                        structures=[],
                        requests=[],
                        repo_root=REPO_ROOT / "hflib" / "transformers",
                        repository={},
                    )

    def test_generate_empty_requests_without_previous_work_dir(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="t13-fresh-") as directory:
            project = Path(directory)
            result = layers.generate_archify_layers(
                REPO_ROOT,
                project_root=project,
                network={},
                structures=[],
                requests=[],
                repo_root=REPO_ROOT / "hflib" / "transformers",
                repository={},
            )
            self.assertEqual(result["manifest"]["entries"], [])
            self.assertTrue((project / ".work").is_dir())

    def test_delivery_identity_is_environment_independent(self):
        """浏览器可用/受限只影响运行证据，不影响 identity/交付字节/产品面板。"""
        from unittest.mock import patch

        def generated(browser_available):
            entry = {
                "slug": "network",
                "level": "network",
                "variant": "network",
                "form": "normalized",
                "title": "fixture",
                "html": "viewer/archify/network.html",
                "html_sha256": "same-html",
                "candidate": "data/archify/network.json",
                "candidate_sha256": "same-candidate",
                "relations": "data/archify/edges/network-relations.json",
                "nodes": 8,
                "connections": 7,
                "relations_count": 7,
                "gates": {
                    "validate": "pass",
                    "deliver": "pass",
                    "check": "pass",
                    "browser-check": "pass" if browser_available else "fail",
                },
                "browser_check_available": browser_available,
                "browser_check_classification": None if browser_available else "environment",
                "links": [],
            }
            manifest = {
                "schema": "fixture",
                "builder": "fixture",
                "tool": {"version": "same"},
                "entries": [entry],
                "files": {},
            }
            return {
                "manifest": manifest,
                "files": {
                    "data/archify/manifest.json": json.dumps(
                        manifest, sort_keys=True
                    ).encode()
                },
                "receipts": {},
                "runtime": {"network": {"browser_check_available": browser_available}},
            }

        bundles = []
        for available in (True, False):
            with patch(
                "structure_reader.extract.deepseek_v4_semantic.build_all_layer_group_bundles",
                return_value={},
            ), patch(
                "structure_reader.render.archify_layers.generate_archify_layers",
                return_value=generated(available),
            ):
                bundles.append(
                    layers_build_archify(
                        REPO_ROOT, REPO_ROOT / "structure_reader", {"variants": []}, {}
                    )
                )
        self.assertEqual(bundles[0]["identity"], bundles[1]["identity"])
        self.assertEqual(bundles[0]["files"], bundles[1]["files"])
        self.assertEqual(bundles[0]["panel"], bundles[1]["panel"])
        delivered = json.loads(
            bundles[0]["files"]["data/archify/manifest.json"].decode("utf-8")
        )
        self.assertNotIn("gates", delivered["entries"][0])
        self.assertNotIn("browser_check_available", delivered["entries"][0])

    def test_single_layer_with_archify_flag_is_rejected(self):
        from structure_reader.build_hf_decoder import HfBuildError, run_hf_build

        with self.assertRaises(HfBuildError) as context:
            run_hf_build(
                REPO_ROOT,
                REPO_ROOT / "structure_reader",
                layer_index=4,
                archify_layers=True,
            )
        self.assertIn("--all-variants", str(context.exception))

    def test_cli_rejects_archify_without_all_variants(self):
        from structure_reader.cli import main

        code = main(
            [
                "build-hf",
                "--model",
                "deepseek_v4_flash_0731_hf",
                "--layer-index",
                "4",
                "--archify-layers",
            ]
        )
        self.assertEqual(code, 2)


class TestRoundTwoSpec(unittest.TestCase):
    """Codex 第一轮需求轴 4 项：外部输入、mHC 出口、SCC 回流、FFN 具名 Linear。"""

    def _block(self):
        return _block_document()

    def test_moe_keeps_external_input_from_outside_view(self):
        from structure_reader.render.diagram import module_groups

        block = self._block()
        graph = layers.build_moe_graph(
            block, modules=module_groups(block), group=GROUP
        )
        inbound = {edge.tensor_id for edge in graph.edges if edge.src == "o-in"}
        self.assertIn("ffn_mhc_collapsed", inbound)

    def test_mhc_keeps_entry_and_returns_three_tensors_to_decoder(self):
        from structure_reader.render.diagram import module_groups

        block = self._block()
        graph = layers.build_mhc_graph(
            block, modules=module_groups(block), group=GROUP
        )
        inbound = {edge.tensor_id for edge in graph.edges if edge.src == "o-in"}
        self.assertIn("hidden_streams_in", inbound)
        consumer = next(
            node for node in graph.nodes if node.node_id == "o-consumer-attn")
        self.assertIn("Decoder.forward", consumer.label)
        returns = {
            edge.tensor_id: edge.label
            for edge in graph.edges if edge.dst == "o-consumer-attn"
        }
        self.assertEqual(returns.get("attn_mhc_collapsed"), "collapsed")
        self.assertEqual(returns.get("attn_mhc_post"), "post")
        self.assertEqual(returns.get("attn_mhc_comb"), "comb")
        # mHC 内部不能出现 Attention/FFN 调用节点。
        self.assertFalse(any(
            node.node_id.startswith("o-o_b_proj")
            or node.node_id.startswith("o-moe_")
            for node in graph.nodes
        ))

    def test_mhc_has_no_hidden_streams_out_main_exit(self):
        from structure_reader.render.diagram import module_groups

        block = self._block()
        graph = layers.build_mhc_graph(
            block, modules=module_groups(block), group=GROUP
        )
        self.assertFalse(any(
            edge.dst == "o-out" or edge.tensor_id == "hidden_streams_out"
            for edge in graph.edges
        ))
        self.assertNotIn("o-out", {node.node_id for node in graph.nodes})

    def test_decoder_modules_are_split_into_acyclic_stages(self):
        from structure_reader.render.diagram import module_groups

        block = _block_document()
        block["operators"].append(
            _op("mask_build", "Mask", ["attn_block_output"], ["mask_ext"])
        )
        block["operators"].append(
            _op("core_consume", "Core", ["attn_block_output", "mask_ext"], ["attn_out"])
        )
        block["tensors"].extend(
            [
                _tensor("mask_ext", ["B", "seqlen"]),
                _tensor("attn_out", ["B", "seqlen", "H"]),
            ]
        )
        graph = layers.build_decoder_graph(
            block, modules=module_groups(block), group=GROUP, attention_types={}
        )
        ranks = layers._ranks(graph)
        adjacency: dict[str, list[str]] = {}
        for edge in graph.edges:
            if edge.kind == "data":
                adjacency.setdefault(edge.src, []).append(edge.dst)
        cyclic = [s for s in layers._strongly_connected(adjacency) if len(s) > 1]
        self.assertEqual(cyclic, [])
        self.assertTrue(ranks)
        self.assertTrue(
            any("阶段" in node.label for node in graph.nodes)
            or len({node.tag for node in graph.nodes}) > 2
        )

    def test_acyclic_guard_rejects_bidirectional_graph(self):
        graph = layers.LevelGraph(key="t", level="decoder", title="t", subtitle="")
        graph.nodes = [
            layers.LevelNode("a", "a", "", "t", "backend", (), "main"),
            layers.LevelNode("b", "b", "", "t", "backend", (), "main"),
        ]
        graph.edges = [
            layers.LevelEdge("a", "b", "t1", "t1", "", "data"),
            layers.LevelEdge("b", "a", "t2", "t2", "", "data"),
        ]
        with self.assertRaises(layers.ArchifyLayersError):
            layers._assert_acyclic(graph)

    def test_ffn_named_linear_steps_project_three_shared_linears(self):
        from structure_reader.extract.deepseek_v4_semantic import ffn_named_linear_steps

        block = _block_document()
        expansions = ffn_named_linear_steps(block)
        applied = [item for item in expansions if item.status == "applied"]
        labels = {
            step.label for item in applied for step in item.steps
        }
        self.assertIn("gate_proj · Linear", labels)
        self.assertIn("up_proj · Linear", labels)
        self.assertIn("down_proj · Linear", labels)
        weights = {
            step.weight["name"]
            for item in applied
            for step in item.steps
            if step.weight
        }
        self.assertIn("shared_experts.gate_proj.weight", weights)
        self.assertIn("shared_experts.down_proj.weight", weights)

    def test_ffn_expansion_rejects_unknown_weight_identity(self):
        from structure_reader.extract.deepseek_v4_semantic import ffn_named_linear_steps

        block = _block_document()
        for op in block["operators"]:
            if op["id"] == "shared_gate_up":
                op["weights"] = ["unknown_weight"]
                op["op"] = "linear+clamp×2"
        expansions = {item.op_id: item for item in ffn_named_linear_steps(block)}
        self.assertEqual(expansions["shared_gate_up"].status, "rejected")
        self.assertIn("gate_proj", expansions["shared_gate_up"].reason)


class TestRoundThreeFfnShapes(unittest.TestCase):
    """Codex 第二轮：FFN 新中间张量的形状/名称/来源与全称展示。"""

    ALIASES = ("'Es'", "'H'", "'2Es'", "['Es'", "['H'", "['2Es'")

    def _expansions(self):
        from structure_reader.extract.deepseek_v4_semantic import ffn_named_linear_steps

        return ffn_named_linear_steps(_block_document())

    def test_intermediate_tensor_facts_are_derived_from_input_and_weight(self):
        facts = [
            fact
            for item in self._expansions()
            if item.status == "applied"
            for step in item.steps
            for fact in step.tensor_facts
        ]
        ids = {fact["id"] for fact in facts}
        self.assertIn("sem:shared_gate_up.gate_proj", ids)
        self.assertIn("sem:shared_gate_up.up_proj", ids)
        self.assertIn("sem:shared_act_down.silu", ids)
        gate = next(fact for fact in facts if fact["id"].endswith("gate_proj"))
        self.assertEqual(gate["shape"], ["B", "seqlen", "Es"])
        self.assertIn("权重输出维", gate["derivation"])
        self.assertTrue(gate["source"])
        silu = next(fact for fact in facts if fact["id"].endswith("silu"))
        self.assertIn("激活", silu["derivation"])

    def test_shape_derivation_marks_unknown_reason(self):
        from structure_reader.extract.deepseek_v4_semantic import _linear_output_shape

        shape, reason = _linear_output_shape(("B", "seqlen", "H"), ("Es",))
        self.assertIsNone(shape)
        self.assertIn("二维", reason)

    def test_moe_intermediate_edges_have_full_name_shapes(self):
        from structure_reader.render.diagram import module_groups

        graph = layers.build_moe_graph(
            _block_document(), modules=module_groups(_block_document()), group=GROUP
        )
        sem_edges = [edge for edge in graph.edges if edge.tensor_id.startswith("sem:")]
        self.assertEqual(len(sem_edges), 3)
        for edge in sem_edges:
            self.assertTrue(edge.shape, edge.tensor_id)
            self.assertIn("batch_size", edge.shape)
            self.assertIn("moe_intermediate_size", edge.shape)
            for alias in ("Es", "2Es"):
                self.assertNotIn(alias, edge.shape)

    def test_ffn_linear_components_use_full_parameter_names(self):
        from structure_reader.render.diagram import module_groups

        graph = layers.build_moe_graph(
            _block_document(), modules=module_groups(_block_document()), group=GROUP
        )
        linear_nodes = [node for node in graph.nodes if "· Linear" in node.label]
        self.assertEqual(len(linear_nodes), 5)
        for node in linear_nodes:
            text = f"{node.label} {node.sublabel}"
            self.assertIn("moe_intermediate_size", text)
            self.assertIn("hidden_size", text)
            for alias in ("'Es'", "'H'", "'2Es'"):
                self.assertNotIn(alias, text)

    def test_shape_validator_rejects_unknown_internal_edge(self):
        graph = layers.LevelGraph(key="t", level="moe", title="t", subtitle="")
        graph.nodes = [
            layers.LevelNode("a", "a", "", "t", "backend", (), "main"),
            layers.LevelNode("b", "b", "", "t", "backend", (), "main"),
        ]
        graph.edges = [layers.LevelEdge("a", "b", "t1", "t1", "", "data")]
        with self.assertRaises(layers.ArchifyLayersError):
            layers.validate_graph_shapes(graph)
        graph.edges = [
            layers.LevelEdge(
                "a", "b", "t1", "t1", "", "data", condition="未知：既有事实未登记形状"
            )
        ]
        layers.validate_graph_shapes(graph)


class TestModelGate(unittest.TestCase):
    def test_unknown_model_key_is_named_rejection(self):
        from structure_reader.build_hf_decoder import HfBuildError, run_hf_build

        with self.assertRaises(HfBuildError) as context:
            run_hf_build(
                REPO_ROOT,
                REPO_ROOT / "structure_reader",
                model_key="not_a_model_key",
                all_variants=True,
                archify_layers=True,
            )
        message = str(context.exception)
        self.assertIn("未知模型键", message)
        self.assertIn("not_a_model_key", message)


@unittest.skipUnless(
    (REPO_ROOT / "hflib/transformers").is_dir(), "缺少只读 HF 检出；如实跳过"
)
class TestToolIdentity(unittest.TestCase):
    def test_identity_has_tool_and_builder(self):
        identity = layers.archify_identity(REPO_ROOT)
        self.assertEqual(identity["builder"], layers.BUILDER_VERSION)
        self.assertEqual(identity["schema"], layers.SCHEMA)
        self.assertTrue(identity.get("cli_sha256"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
