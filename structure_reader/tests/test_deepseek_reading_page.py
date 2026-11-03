"""D01：DeepSeek 正式阅读入口测试（整网 + 端点逐算子 + 四层导航 + 两阶段）。

覆盖的是用户可见行为，不是实现镜像：
- 整网简单端点的 Embedding / 输出端必须是具名算子的逐算子图，HyperHead 不能是单节点；
- 图文字完整（候选里不出现省略号），完整文字/逐条关系在线下可读；
- 页面只有 Prefill / Decode 两个阶段，不出现第三类数值场景；
- 四层路由、内嵌原生图、单独打开、参数/源码联查都离线可用。

运行：cd structure_reader && PYTHONPATH=src python3 tests/test_deepseek_reading_page.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from structure_reader.extract.deepseek_v4_network import build_network_overview  # noqa: E402
from structure_reader.render import archify_layers as layers  # noqa: E402
from structure_reader.render.hf_reading_page import (  # noqa: E402
    ReadingPageError,
    render_reading_page,
)

REPOSITORY = {
    "url": "https://github.com/huggingface/transformers.git",
    "revision": "536ecc007387a50e77603bb5d92100e9b07514cc",
    "link_mode": "local-only",
}

_READY = (REPO_ROOT / "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json").is_file()
_skip = unittest.skipUnless(_READY, "缺少只读 HF 输入；如实跳过")


def _network() -> dict:
    return build_network_overview(REPO_ROOT, model_key="deepseek_v4_flash_0731_hf")


def _items() -> list[dict]:
    from structure_reader.extract.hf_layer_structures import build_structures_container

    container = build_structures_container(REPO_ROOT)
    return [
        {key: item[key] for key in (
            "id", "label", "attention_type", "moe_type", "layer_indices",
            "representative_layer_index", "structure")}
        for item in container["variants"]
    ]


def _candidate(graph) -> dict:
    return layers.build_level_candidate(
        graph,
        output_relpath=f"viewer/archify/{graph.key}.html",
        repository=REPOSITORY,
        tool=None,
        quality="standard",
        complete_text=True,
    )


def _entry(slug: str, title: str, candidate: dict, relations: list[dict]) -> dict:
    components = candidate["components"]
    return {
        "slug": slug,
        "title": title,
        "html": f"viewer/archify/{slug}.html",
        "view": f"viewer/archify/{slug}.view.html",
        "candidate": f"data/archify/{slug}.json",
        "relations": f"data/archify/edges/{slug}-relations.json",
        "nodes": len(components),
        "connections": len(candidate["connections"]),
        "relations_count": len(relations),
        "node_details": [
            {
                "id": comp["id"],
                "label": comp["label"],
                "sublabel": comp.get("sublabel", ""),
                "tag": comp.get("tag", ""),
                "sources": comp.get("sources", []),
            }
            for comp in components
        ],
        "relations_all": relations,
        "canvas": {
            "width": max(c["pos"][0] + c["size"][0] for c in components) + 60,
            "height": max(c["pos"][1] + c["size"][1] for c in components) + 70,
        },
    }


def _panel() -> dict:
    network = _network()
    items = _items()
    network_graph = layers.build_network_graph(network)
    embed_graph = layers.build_endpoint_graph(network, "embedding")
    head_graph = layers.build_endpoint_graph(network, "head")
    groups = {}
    for item in items:
        gid = str(item["id"])
        block = item["structure"]["block"]
        modules = layers.module_groups(block)
        group = next(g for g in network["groups"] if str(g["id"]) == gid)
        decoder = layers.build_decoder_graph(block, modules=modules, group=group,
                                             attention_types={})
        decoder_entry = _entry(f"decoder-{gid}", decoder.title,
                               _candidate(decoder), decoder.relations())
        mhc = layers.build_mhc_graph(block, modules=modules, group=group)
        moe = layers.build_moe_graph(block, modules=modules, group=group)
        groups[gid] = {
            "group": None,
            "decoder": decoder_entry,
            # 这里只验证页签结构与作用域：attention/kv 用同一份候选占位，
            # 真实语义子图由完整构建提供（D01-verify 绑定实际候选）。
            "kv": [dict(decoder_entry, slug=f"kv-{gid}", form="normalized")],
            "attention": [dict(decoder_entry, slug=f"attention-{gid}",
                               form="normalized")],
            "moe": _entry(f"moe-{gid}", moe.title, _candidate(moe), moe.relations()),
            "mhc": _entry(f"mhc-{gid}", mhc.title, _candidate(mhc), mhc.relations()),
        }
    return {
        "schema": "hf-archify-layers.v1",
        "network": _entry("network", network_graph.title, _candidate(network_graph),
                          network_graph.relations()),
        "endpoints": {
            "embedding": _entry("embedding", embed_graph.title,
                                _candidate(embed_graph), embed_graph.relations()),
            "head": _entry("head", head_graph.title, _candidate(head_graph),
                           head_graph.relations()),
        },
        "groups": groups,
        "tool": {"version": "3.2.0", "node": "v26.10.0", "cli_sha256": "0" * 64},
    }


def _html() -> str:
    return render_reading_page(
        _network(), _items(), candidate_id="test-candidate", archify=_panel(),
    ).decode("utf-8")


@_skip
class TestEndpointOperatorGraphs(unittest.TestCase):
    """Embedding / 输出端从整网同源事实展开为具名算子。"""

    @classmethod
    def setUpClass(cls):
        cls.network = _network()

    def test_embedding_is_gather_plus_multistream(self):
        graph = layers.build_endpoint_graph(self.network, "embedding")
        labels = {node.node_id: node.label for node in graph.nodes}
        self.assertIn("e-net_embed", {n.node_id for n in graph.nodes})
        self.assertIn("e-net_hc_expand", {n.node_id for n in graph.nodes})
        self.assertIn("input_ids", " ".join(labels.values()))
        self.assertIn("hidden_streams", " ".join(labels.values()))
        # 权重张量就近出现，带真实形状
        weight = next(n for n in graph.nodes if n.node_id == "e-embed_tokens_weight")
        self.assertEqual(weight.role, "aux")
        self.assertIn("vocab_size", weight.sublabel)

    def test_head_expands_hyperhead_not_single_node(self):
        graph = layers.build_endpoint_graph(self.network, "head")
        op_ids = {op for node in graph.nodes for op in node.op_ids}
        for expected in ("net_hc_head_norm", "net_hc_head_project",
                         "net_hc_head_pre", "net_hc_head_collapse",
                         "net_final_norm", "net_logits_slice", "net_lm_head"):
            self.assertIn(expected, op_ids)
        hyper = [node for node in graph.nodes if node.op_ids
                 and node.op_ids[0].startswith("net_hc_head_")]
        self.assertGreaterEqual(len(hyper), 4)
        # 门控权重/偏置就近出现
        aux_tags = {node.node_id for node in graph.nodes if node.role == "aux"}
        self.assertIn("e-hc_head_fn", aux_tags)
        self.assertIn("e-hc_head_base", aux_tags)
        self.assertIn("e-hc_head_scale", aux_tags)
        self.assertIn("e-final_norm_weight", aux_tags)
        self.assertIn("e-lm_head_weight", aux_tags)

    def test_head_output_uses_logits_to_keep_symbol(self):
        graph = layers.build_endpoint_graph(self.network, "head")
        params = {item["id"]: item for item in graph.parameter_items}
        self.assertIn("Lk", params)
        self.assertIn("Ssel", params)
        self.assertIn("min(seqlen, Lk)", str(params["Ssel"].get("expression")))
        logits = next(node for node in graph.nodes if node.node_id == "out-logits")
        self.assertIn("logits_sequence_length", logits.sublabel)

    def test_candidates_have_no_truncated_main_text(self):
        for kind in ("embedding", "head"):
            graph = layers.build_endpoint_graph(self.network, kind)
            candidate = _candidate(graph)
            for component in candidate["components"]:
                for field in ("label", "sublabel", "tag"):
                    self.assertNotIn("…", component.get(field) or "",
                                     f"{kind}:{component['id']}:{field}")
            for connection in candidate["connections"]:
                self.assertNotIn("…", str(connection.get("label") or ""))

    def test_every_weight_node_has_a_consumer_edge(self):
        """权重节点不能只有节点没有消费边；按 (权重, 消费算子) 保留。"""
        expectations = {
            "embedding": {"e-embed_tokens_weight": "e-net_embed"},
            "head": {
                "e-hc_head_fn": "e-net_hc_head_project",
                "e-hc_head_base": "e-net_hc_head_pre",
                "e-hc_head_scale": "e-net_hc_head_pre",
                "e-final_norm_weight": "e-net_final_norm",
                "e-lm_head_weight": "e-net_lm_head",
            },
        }
        for kind, expected in expectations.items():
            graph = layers.build_endpoint_graph(self.network, kind)
            weights = {node.node_id for node in graph.nodes
                       if node.role == "aux" and node.sublabel}
            self.assertTrue(weights, kind)
            for weight_id, consumer_id in expected.items():
                outgoing = [edge for edge in graph.edges if edge.src == weight_id]
                self.assertTrue(
                    outgoing,
                    f"{kind}: {weight_id} 没有任何消费连线（权重依赖丢失）",
                )
                self.assertIn(
                    consumer_id, {edge.dst for edge in outgoing},
                    f"{kind}: {weight_id} 未连到真实消费者 {consumer_id}",
                )
                for node_id in weights:
                    incident = [edge for edge in graph.edges
                                if edge.src == node_id or edge.dst == node_id]
                    self.assertTrue(incident, f"{kind}: {node_id} 无边")

    def test_weight_sources_use_consumer_or_producer_not_first_op(self):
        """外部/权重来源必须指向真实定义/消费位置，不能统一继承首个算子。"""
        graph = layers.build_endpoint_graph(self.network, "head")
        by_id = {node.node_id: node for node in graph.nodes}
        for weight_id, line in (
            ("e-hc_head_fn", 995),          # 定义
            ("e-hc_head_base", 1002),       # 消费位置（pre 公式）
            ("e-hc_head_scale", 1002),      # 消费位置（pre 公式）
            ("e-final_norm_weight", 1307),  # 定义
            ("e-lm_head_weight", 1451),     # 定义
        ):
            lines = {source["line"] for source in by_id[weight_id].sources}
            self.assertIn(line, lines, f"{weight_id} 来源未指向定义/消费位置 {line}")
        # 输入端点：input_ids 由消费算子给出消费位置
        embedding = layers.build_endpoint_graph(self.network, "embedding")
        input_ids = next(n for n in embedding.nodes if n.node_id == "e-input_ids")
        self.assertTrue(input_ids.sources)
        self.assertIn("消费算子", " ".join(input_ids.details))
        # 上游张量使用真实生产算子（layer loop），而不是端点首算子
        streams = layers.build_endpoint_graph(self.network, "head").node("e-streams_after_layers")
        self.assertIn("net_layer_loop", " ".join(streams.details))


@_skip
class TestReadingPage(unittest.TestCase):
    """正式入口只显示 Archify 图，且两阶段 / 四层 / 离线联查齐备。"""

    @classmethod
    def setUpClass(cls):
        cls.html = _html()
        cls.items = _items()

    def test_only_prefill_and_decode_stages(self):
        h = self.html
        self.assertEqual(h.count("data-stage-btn="), 2)
        self.assertIn('data-stage-btn="prefill"', h)
        self.assertIn('data-stage-btn="decode"', h)
        # 旧页的 5 类数值场景不得再出现在阅读 UI
        for forbidden in ("prefill-empty", "prefill-symbolic", "decode-nonzero",
                          "decode-closed", "decode-multi", 'id="scn-data"'):
            self.assertNotIn(forbidden, h)
        self.assertIn("不代入示例长度", h)

    def test_four_layers_are_addressable(self):
        h = self.html
        self.assertIn('id="lv-network"', h)
        self.assertIn('href="#/network"', h)
        for item in self.items:
            gid = item["id"]
            for level in ("group", "block", "ops"):
                self.assertIn(f'id="lv-{level}-{gid}"', h)
            self.assertIn(f'href="#/group/{gid}"', h)
            self.assertIn(f'href="#/group/{gid}/block"', h)
            self.assertIn(f'href="#/group/{gid}/ops"', h)

    def test_graphs_are_embedded_native_archify(self):
        h = self.html
        iframes = re.findall(r'<iframe class="rd-frame" src="([^"]+)"', h)
        self.assertGreaterEqual(len(iframes), 3)  # 整网 + 2 端点 + 每组 Decoder
        for src in iframes:
            self.assertIn("archify/", src)
            self.assertIn("embed=1", src)
        # 每张图都有原生单开入口，而不是只给内部 API
        self.assertIn("原生图新标签", h)
        self.assertIn("子图页（带返回）", h)

    def test_full_text_and_relations_are_complete(self):
        h = self.html
        self.assertIn("节点完整名称与操作", h)
        self.assertIn("数据流来源与目标", h)
        self.assertIn("HyperHead：流展平+归一", h)
        self.assertIn("self.hc_fn", h)
        # 完整文字面板里不能出现被截断的主图标签
        for block in re.findall(r'<p class="rd-full-node">(.*?)</p>', h, re.S):
            self.assertNotIn("…", block)
        # 逐条关系表按实际关系数渲染，不截前 N 条
        self.assertIn("条关系", h)

    def test_parameters_and_source_lookup_are_offline(self):
        h = self.html
        self.assertIn('id="p-network-Lk"', h)
        self.assertIn('id="p-network-Ssel"', h)
        self.assertIn("toggleSource", h)  # 源码快照面板开关
        self.assertIn("reports/source_manifest.json", h)
        self.assertIn("data/archify/", h)

    def test_scope_banner_declares_undelivered_layers(self):
        h = self.html
        self.assertIn("本票（D04）", h)
        self.assertIn("D03", h)
        self.assertIn("D04", h)
        self.assertIn("尚未交付", h)
        self.assertNotIn("本票（D01）交付", h)

    def test_internal_anchors_resolve(self):
        ids = set(re.findall(r'id="([^"]+)"', self.html))
        anchors = [href for href in re.findall(r'href="#([^"]+)"', self.html)
                   if not href.startswith("/")]
        broken = sorted({anchor for anchor in anchors if anchor not in ids})
        self.assertEqual(broken, [], f"悬空锚点: {broken[:10]}")

    def test_endpoint_nearby_parameters_are_rendered(self):
        h = self.html
        for param in ("Lk", "Ssel", "Nhc", "Vocab"):
            self.assertIn(f'data-param="{param}"', h)
        # 输出端必须有 logits_to_keep 的符号参数，而不是写死长度
        self.assertIn("logits_to_keep", h)

    def test_fulltext_carries_source_locations(self):
        h = self.html
        self.assertIn("来源：", h)
        self.assertIn("modeling_deepseek_v4.py", h)

    def test_rendering_requires_archify_panel(self):
        with self.assertRaises(ReadingPageError):
            render_reading_page(_network(), _items(), archify=None)

    def test_rendered_page_is_deterministic(self):
        self.assertEqual(_html(), _html())


@_skip
class TestPublicGraphText(unittest.TestCase):
    """AC14：本票公开的 Decoder / mHC / MoE 图主图也必须完整，不靠文字表兜底。"""

    @classmethod
    def setUpClass(cls):
        cls.network = _network()
        cls.items = _items()

    def _candidates(self):
        for item in self.items:
            gid = item["id"]
            block = item["structure"]["block"]
            modules = layers.module_groups(block)
            group = next(g for g in self.network["groups"] if str(g["id"]) == gid)
            yield gid, "decoder", layers.build_decoder_graph(
                block, modules=modules, group=group, attention_types={})
            yield gid, "mhc", layers.build_mhc_graph(
                block, modules=modules, group=group)
            yield gid, "moe", layers.build_moe_graph(
                block, modules=modules, group=group)

    def test_public_graphs_have_no_truncated_main_text(self):
        checked = 0
        for gid, kind, graph in self._candidates():
            candidate = layers.build_level_candidate(
                graph, output_relpath=f"viewer/archify/{kind}-{gid}.html",
                repository=REPOSITORY, tool=None, quality="standard",
                complete_text=True,
            )
            for component in candidate["components"]:
                for field in ("label", "sublabel", "tag"):
                    value = component.get(field) or ""
                    self.assertNotIn(
                        "…", value, f"{kind}-{gid}:{component['id']}:{field}={value!r}")
            for connection in candidate["connections"]:
                self.assertNotIn(
                    "…", str(connection.get("label") or ""),
                    f"{kind}-{gid}:{connection['id']} 连线标签被截断")
            checked += 1
        self.assertEqual(checked, 12)

    def test_complete_text_is_wider_but_bounded(self):
        for _gid, kind, graph in self._candidates():
            candidate = layers.build_level_candidate(
                graph, output_relpath="viewer/archify/x.html",
                repository=REPOSITORY, tool=None, quality="standard",
                complete_text=True,
            )
            widths = {component["size"][0] for component in candidate["components"]}
            self.assertTrue(widths)
            self.assertLessEqual(max(widths), 760,
                                 f"{kind}: 节点宽度不应无限放大")


NODE = Path("/opt/homebrew/bin/node")


@_skip
class TestTabGroups(unittest.TestCase):
    """AC05：页签组必须按章节隔离，跨层往返不能把其他章节的图全部隐藏。"""

    @classmethod
    def setUpClass(cls):
        cls.html = _html()

    def test_every_tab_lives_in_a_named_group(self):
        groups = re.findall(r'data-tab-group="([^"]+)"', self.html)
        self.assertEqual(len(groups), 9, groups)  # 整网 1 + 每组 block/ops 各 1
        self.assertIn("network-endpoints", groups)
        for gid in ("swa-hash", "csa-hash", "hca-topk", "csa-topk"):
            self.assertIn(f"block-{gid}", groups)
            self.assertIn(f"ops-{gid}", groups)
        # 页签操作必须落在组内；整页 [data-tab] 选择器已经不存在
        self.assertNotIn(
            'document.querySelectorAll("[data-tab]")', self.html,
            "页签切换仍在操作整页所有 [data-tab]")
        self.assertIn('group.querySelectorAll("[data-tab]")', self.html)
        self.assertIn('closest("[data-tab-group]")', self.html)
        self.assertEqual(len(re.findall(r'data-tab-btn="', self.html)), 22)
        # 跨层进入时必须恢复该层有效页签，而不是留下空白
        self.assertIn("ensureActiveTabs(document.getElementById(activeId))",
                      self.html)

    @unittest.skipUnless(NODE.is_file(), "本机无 node；跳过 JS 作用域回归")
    def test_actual_bindTabs_isolates_groups_and_reveal(self):
        result = _run_tab_scope_probe(self.html)

        def visible(group):
            return {key for key, hidden in result[group] if not hidden}

        def state(stage, group):
            return {key for key, hidden in result[stage][group] if not hidden}

        self.assertEqual(state("initial", "g1"), {"embedding"})
        # 点击整网 head：只有整网组切换，block/ops 组保持原选中
        self.assertEqual(state("afterHead", "g1"), {"head"})
        self.assertEqual(state("afterHead", "g2"), state("initial", "g2"))
        self.assertEqual(state("afterHead", "g3"), state("initial", "g3"))
        # 点击 block 页签：只切本组，整网 head 与 ops 组保持
        self.assertEqual(state("afterBlock", "g2"), {"block-attention"})
        self.assertEqual(state("afterBlock", "g1"), {"head"})
        self.assertEqual(state("afterBlock", "g3"), state("initial", "g3"))
        # reveal 只激活自身所在组
        self.assertEqual(state("afterReveal", "g3"), {"ops-kv"})
        self.assertEqual(state("afterReveal", "g1"), {"head"})
        self.assertEqual(state("afterReveal", "g2"), {"block-attention"})
        # 组内全部隐藏时 ensureActiveTabs 恢复第一张，而不是空白
        self.assertEqual(state("afterEnsure", "g1"), {"embedding"})
        # 跨层往返：进入 block 再回整网，整网仍选中 head；block 保持自己的页签
        self.assertEqual(result["inBlock"]["active"], ["lv-block-swa-hash"])
        self.assertEqual(state("inBlock", "g2"), {"block-attention"})
        self.assertEqual(state("inBlock", "g1"), {"head"})
        self.assertEqual(result["backNetwork"]["active"], ["lv-network"])
        self.assertEqual(state("backNetwork", "g1"), {"head"})
        # 进入某层时若该组无可见图，show() 恢复第一张
        self.assertEqual(state("repairedBlock", "g2"), {"block-mhc"})


def _run_tab_scope_probe(html: str) -> dict:
    start = html.index("function esc(")
    end = html.index("window.locateInDiagram")
    source = html[start:end]
    probe = r"""const vm = require('node:vm');
const source = __SOURCE__;
function makeGroup(name, keys, active) {
  const group = { name: name, bodies: [], buttons: [] };
  keys.forEach(function (key) {
    const body = {
      key: key, hidden: key !== active, parentElement: null, tagName: 'DIV',
      getAttribute: function (n) { return n === 'data-tab' ? key : null; },
      closest: function (sel) {
        if (sel === '[data-tab]') return this;
        if (sel === '[data-tab-group]') return group;
        return null;
      },
      scrollIntoView: function () {},
      classList: { add: function () {}, remove: function () {} }
    };
    const button = {
      key: key, clickHandler: null,
      getAttribute: function (n) { return n === 'data-tab-btn' ? key : null; },
      closest: function (sel) {
        return sel === '[data-tab-group]' ? group : null;
      },
      addEventListener: function (type, fn) {
        if (type === 'click') button.clickHandler = fn;
      },
      click: function () { if (button.clickHandler) button.clickHandler(); },
      classList: { on: false, toggle: function (_, on) { this.on = on; } }
    };
    group.bodies.push(body);
    group.buttons.push(button);
  });
  group.querySelectorAll = function (sel) {
    if (sel === '[data-tab]') return group.bodies;
    if (sel === '[data-tab-btn]') return group.buttons;
    if (sel === '[data-tab-group]') return [group];
    return [];
  };
  group.closest = function (sel) {
    return sel === '[data-tab-group]' ? group : null;
  };
  return group;
}
const g1 = makeGroup('network-endpoints', ['embedding', 'head'], 'embedding');
const g2 = makeGroup('block-swa-hash',
  ['block-mhc', 'block-attention', 'block-moe'], 'block-mhc');
const g3 = makeGroup('ops-swa-hash', ['ops-attention', 'ops-kv'], 'ops-attention');
function makeSection(id, groups) {
  return {
    id: id, hidden: id !== 'lv-network', style: {},
    querySelectorAll: function (sel) {
      return sel === '[data-tab-group]' ? groups : [];
    },
    getAttribute: function (n) { return n === 'data-lv' ? id.replace('lv-', '') : null; }
  };
}
const sections = [
  makeSection('lv-network', [g1]),
  makeSection('lv-block-swa-hash', [g2]),
  makeSection('lv-ops-swa-hash', [g3])
];
const byId = {};
[g1, g2, g3].forEach(function (g) {
  g.bodies.forEach(function (b) { byId[b.key] = b; });
});
const crumbs = { innerHTML: '' };
const document = {
  querySelectorAll: function (sel) {
    if (sel === '[data-tab-btn]') return [].concat(g1.buttons, g2.buttons, g3.buttons);
    if (sel === '[data-tab]') return [].concat(g1.bodies, g2.bodies, g3.bodies);
    if (sel === '[data-tab-group]') return [g1, g2, g3];
    return [];
  },
  getElementById: function (id) {
    if (byId[id]) return byId[id];
    if (id === 'rd-crumbs') return crumbs;
    for (let i = 0; i < sections.length; i += 1) {
      if (sections[i].id === id) return sections[i];
    }
    return null;
  },
  addEventListener: function () {},
  body: {
    setAttribute: function () {}, getAttribute: function () { return ''; }
  },
  readyState: 'loading'
};
const window = {
  addEventListener: function () {}, location: { hash: '' }, scrollTo: function () {}
};
const context = {
  console: console, document: document, window: window, sections: sections,
  GROUP_LABELS: {
    'swa-hash': 'SWA + Hash MoE', 'csa-hash': 'CSA + Hash MoE',
    'hca-topk': 'HCA + TopK MoE', 'csa-topk': 'CSA + TopK MoE'
  },
  state: { stage: 'prefill' }, setTimeout: function () {}
};
vm.runInNewContext(source + '\nbindTabs();', context);
function snapshot(g) {
  return g.bodies.map(function (b) { return [b.key, b.hidden]; });
}
const initial = { g1: snapshot(g1), g2: snapshot(g2), g3: snapshot(g3) };
g1.buttons[1].click();
const afterHead = { g1: snapshot(g1), g2: snapshot(g2), g3: snapshot(g3) };
g2.buttons[1].click();
const afterBlock = { g1: snapshot(g1), g2: snapshot(g2), g3: snapshot(g3) };
context.reveal('ops-kv');
const afterReveal = { g1: snapshot(g1), g2: snapshot(g2), g3: snapshot(g3) };
g1.bodies.forEach(function (b) { b.hidden = true; });
g1.buttons.forEach(function (b) { b.classList.on = false; });
context.ensureActiveTabs(g1);
const afterEnsure = { g1: snapshot(g1), g2: snapshot(g2), g3: snapshot(g3) };
// 跨层往返：先重新选中整网 head，再进入 block、返回整网。
g1.buttons[1].click();
context.show({ level: 'block', gid: 'swa-hash' });
const inBlock = { g1: snapshot(g1), g2: snapshot(g2), g3: snapshot(g3),
  active: sections.filter(function (s) { return !s.hidden; }).map(function (s) { return s.id; }) };
context.show({ level: 'network' });
const backNetwork = { g1: snapshot(g1), g2: snapshot(g2), g3: snapshot(g3),
  active: sections.filter(function (s) { return !s.hidden; }).map(function (s) { return s.id; }) };
// 进入 block 前把该组全部隐藏：show() 应恢复第一张，而不是空白。
g2.bodies.forEach(function (b) { b.hidden = true; });
context.show({ level: 'block', gid: 'swa-hash' });
const repairedBlock = { g1: snapshot(g1), g2: snapshot(g2), g3: snapshot(g3) };
console.log(JSON.stringify({ initial: initial, afterHead: afterHead,
  afterBlock: afterBlock, afterReveal: afterReveal, afterEnsure: afterEnsure,
  inBlock: inBlock, backNetwork: backNetwork, repairedBlock: repairedBlock }));
""".replace("__SOURCE__", json.dumps(source))
    with tempfile.TemporaryDirectory(prefix="d01-tab-scope-") as work:
        script = Path(work) / "probe.cjs"
        script.write_text(probe, encoding="utf-8")
        run = subprocess.run(
            [str(NODE), str(script)], capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        raise AssertionError(f"页签作用域探针失败:\n{run.stderr[:2000]}")
    return json.loads(run.stdout)


@_skip
class TestNetworkLayout(unittest.TestCase):
    """整网主链单列、辅助预计算在左旁枝；全部节点/边/张量身份保留。"""

    @classmethod
    def setUpClass(cls):
        cls.network = _network()
        cls.graph = layers.build_network_graph(cls.network)
        cls.candidate = layers.build_level_candidate(
            cls.graph, output_relpath="viewer/archify/network.html",
            repository=REPOSITORY, tool=None, quality="standard",
            complete_text=True,
        )

    def test_main_chain_is_one_column_and_canvas_is_compact(self):
        components = self.candidate["components"]
        by_id = {comp["id"]: comp for comp in components}
        # token→logits 主链（mask 是 rank 2 的第二列分支，不算主链）
        chain_ids = [
            node.node_id for node in self.graph.nodes
            if node.role in ("boundary_in", "main", "boundary_out")
            and node.node_id != "n-net_mask_build"
        ]
        xs = {by_id[layers._component_id(node_id)]["pos"][0] for node_id in chain_ids}
        self.assertEqual(len(xs), 1, "主链应保持单列，避免下半部偏到最右列")
        width = max(comp["pos"][0] + comp["size"][0] for comp in components)
        self.assertLessEqual(width, 1200, "整网画布不应再因四列并排而过宽")

    def test_precompute_nodes_are_side_branches_with_anchors(self):
        by_id = {node.node_id: node for node in self.graph.nodes}
        for node_id, anchor in (
            ("n-net_position_ids", "n-input"),
            ("n-net_rope_main", "n-net_embed"),
            ("n-net_rope_compress", "n-net_embed"),
        ):
            self.assertEqual(by_id[node_id].role, "branch")
            self.assertEqual(by_id[node_id].anchor, anchor)
        # mask 构建是 rank 2 的第二列分支，不是主链铺开
        self.assertEqual(by_id["n-net_mask_build"].role, "main")

    def test_all_network_edges_and_tensor_shapes_are_retained(self):
        self.assertEqual(len(self.graph.nodes), 19)
        self.assertEqual(len(self.graph.edges), 27)
        masked = [edge for edge in self.graph.edges
                  if edge.dst in ("n-net_mask_build", "n-net_rope_main",
                                  "n-net_rope_compress")]
        self.assertTrue(masked)
        for edge in masked:
            self.assertTrue(edge.tensor_id)
            self.assertTrue(edge.tensor_label)
            self.assertTrue(edge.shape, f"{edge.src}->{edge.dst} 丢失 shape")

    def test_side_inputs_flow_downward(self):
        """侧栏辅助输入到消费者的数据边不应向上回指（state 读写除外）。"""
        positions = {
            comp["id"]: (comp["pos"][0], comp["pos"][1])
            for comp in self.candidate["components"]
        }
        for edge in self.graph.edges:
            if edge.kind != "data":
                continue
            src = positions[layers._component_id(edge.src)]
            dst = positions[layers._component_id(edge.dst)]
            self.assertGreaterEqual(
                dst[1], src[1],
                f"{edge.src}->{edge.dst} 数据边向上回指",
            )

    def test_network_candidate_has_no_truncated_text(self):
        for component in self.candidate["components"]:
            for field in ("label", "sublabel", "tag"):
                self.assertNotIn("…", component.get(field) or "")
        for connection in self.candidate["connections"]:
            self.assertNotIn("…", str(connection.get("label") or ""))

    def test_previously_failing_edge_keeps_full_shape_text(self):
        """首轮失败边 embed→mask 仍保留全称 shape，且在侧栏空白可读位置。"""
        edge = next(
            conn for conn in self.candidate["connections"]
            if conn["from"] == "n-n-net_embed"
            and conn["to"] == "n-n-net_mask_build"
        )
        self.assertEqual(edge["label"], "batch_size,seqlen,hidden_size")
        canvas_width = max(
            comp["pos"][0] + comp["size"][0] for comp in self.candidate["components"])
        self.assertGreaterEqual(edge["labelAt"][0], 0)
        self.assertLessEqual(edge["labelAt"][0], canvas_width)

    def test_side_labels_do_not_overlap_each_other(self):
        labelled = []
        for conn in self.candidate["connections"]:
            text = conn.get("label")
            if not text or not conn.get("labelAt"):
                continue
            width = layers._edge_label_width(text)
            cx, cy = conn["labelAt"]
            labelled.append((cx - width / 2, cy - 9, width, 10, text))
        for index, left in enumerate(labelled):
            for right in labelled[index + 1:]:
                overlap = not (
                    left[0] + left[2] <= right[0]
                    or left[0] >= right[0] + right[2]
                    or left[1] + left[3] <= right[1]
                    or left[1] >= right[1] + right[3]
                )
                self.assertFalse(
                    overlap,
                    f"连线标签重叠: {left[4]!r} 与 {right[4]!r}")


class TestEdgeLabelLayout(unittest.TestCase):
    """平行边、汇入边、跨层长标签的共用去重叠回归。"""

    @classmethod
    def setUpClass(cls):
        network = _network()
        cls.network = network
        variants = _items()
        graphs = {
            "network": layers.build_network_graph(network),
            "head": layers.build_endpoint_graph(network, "head"),
        }
        for item in variants:
            group_id = item["id"]
            block = item["structure"]["block"]
            modules = layers.module_groups(block)
            group = next(
                entry for entry in network["groups"] if entry["id"] == group_id)
            graphs[f"decoder-{group_id}"] = layers.build_decoder_graph(
                block, modules=modules, group=group, attention_types={})
            graphs[f"mhc-{group_id}"] = layers.build_mhc_graph(
                block, modules=modules, group=group)
            graphs[f"moe-{group_id}"] = layers.build_moe_graph(
                block, modules=modules, group=group)
        cls.candidates = {}
        for slug, graph in graphs.items():
            cls.candidates[slug] = layers.build_level_candidate(
                graph, output_relpath=f"viewer/archify/{slug}.html",
                repository=REPOSITORY, tool=None, quality="standard",
                complete_text=True,
            )

    @staticmethod
    def _label_rects(candidate):
        rects = []
        for conn in candidate["connections"]:
            text = conn.get("label")
            at = conn.get("labelAt")
            if not text or not at:
                continue
            width = layers._edge_label_width(text)
            rects.append((at[0] - width / 2, at[1] - 9, width, 10,
                          conn["id"], text))
        return rects

    def test_no_two_edge_labels_overlap(self):
        for slug, candidate in self.candidates.items():
            rects = self._label_rects(candidate)
            for index, left in enumerate(rects):
                for right in rects[index + 1:]:
                    self.assertFalse(
                        layers._rects_overlap(left, right, 1.0),
                        f"{slug}: {left[4]} {left[5]!r} 与 "
                        f"{right[4]} {right[5]!r} 重叠",
                    )

    def test_no_edge_label_covers_a_node(self):
        for slug, candidate in self.candidates.items():
            nodes = [
                (comp["pos"][0], comp["pos"][1],
                 comp["size"][0], comp["size"][1])
                for comp in candidate["components"]
            ]
            for label in self._label_rects(candidate):
                for node in nodes:
                    self.assertFalse(
                        layers._rects_overlap(label, node, 1.0),
                        f"{slug}: 标签 {label[5]!r} 覆盖节点",
                    )

    def test_no_edge_label_covers_node_title_text(self):
        """完整文字模式节点标题可能比节点框更宽，标签不能压住标题。"""
        for slug, candidate in self.candidates.items():
            for label in self._label_rects(candidate):
                for comp in candidate["components"]:
                    text_width = max(
                        layers._edge_label_width(comp["label"]) * 11 / 8,
                        layers._edge_label_width(comp.get("sublabel") or "") * 9 / 8,
                    )
                    expand = max(0.0, (text_width - comp["size"][0]) / 2)
                    title_box = (
                        comp["pos"][0] - expand, comp["pos"][1],
                        comp["size"][0] + expand * 2, comp["size"][1],
                    )
                    self.assertFalse(
                        layers._rects_overlap(label, title_box, 1.0),
                        f"{slug}: 标签 {label[5]!r} 压住节点标题 {comp['label']!r}",
                    )

    def test_parallel_labels_are_staggered(self):
        candidate = self.candidates["moe-swa-hash"]
        conns = {
            conn["id"]: conn for conn in candidate["connections"]
            if conn["from"] == "n-o-moe_hash_lookup"
        }
        a = conns["f-5-o-moe-hash-lookup-o-moe-router"]
        b = conns["f-6-o-moe-hash-lookup-o-moe-dispatch"]
        self.assertEqual(a["label"], b["label"])
        self.assertNotEqual(a["labelAt"], b["labelAt"])

    def test_converging_labels_are_staggered(self):
        candidate = self.candidates["decoder-csa-hash"]
        a = next(conn for conn in candidate["connections"]
                 if conn["id"] == "f-14-m-attn-q-2-m-attn-core-9")
        b = next(conn for conn in candidate["connections"]
                 if conn["id"] == "f-15-m-attn-mask-8-m-attn-core-9")
        self.assertNotEqual(a["labelAt"], b["labelAt"])
        self.assertGreaterEqual(abs(a["labelAt"][1] - b["labelAt"][1]), 14)

    def test_long_labels_stay_bounded_and_separated(self):
        candidate = self.candidates["decoder-csa-hash"]
        bottom = [
            conn for conn in candidate["connections"]
            if conn["to"] == "n-d-out" and conn.get("labelAt")
        ]
        self.assertGreaterEqual(len(bottom), 4)
        for conn in bottom:
            left = conn["labelAt"][0] - layers._edge_label_width(conn["label"]) / 2
            self.assertGreaterEqual(left, 0)
        for index, left in enumerate(bottom):
            for right in bottom[index + 1:]:
                self.assertFalse(
                    layers._rects_overlap(
                        (left["labelAt"][0] - layers._edge_label_width(left["label"]) / 2,
                         left["labelAt"][1] - 9,
                         layers._edge_label_width(left["label"]), 10),
                        (right["labelAt"][0] - layers._edge_label_width(right["label"]) / 2,
                         right["labelAt"][1] - 9,
                         layers._edge_label_width(right["label"]), 10),
                        1.0,
                    ),
                    "跨层长标签在底部仍重叠",
                )


class TestD02DecoderMhc(unittest.TestCase):
    """D02：Decoder 两段 mHC 分支与 mHC 权重细图。"""

    @classmethod
    def setUpClass(cls):
        network = _network()
        cls.variants = _items()
        cls.decoders = {}
        cls.mhc = {}
        cls.mhc_graphs = {}
        for item in cls.variants:
            block = item["structure"]["block"]
            modules = layers.module_groups(block)
            group = next(
                entry for entry in network["groups"] if entry["id"] == item["id"])
            cls.decoders[item["id"]] = _candidate(
                layers.build_decoder_graph(
                    block, modules=modules, group=group, attention_types={})
            )
            mhc_graph = layers.build_mhc_graph(block, modules=modules, group=group)
            cls.mhc_graphs[item["id"]] = mhc_graph
            cls.mhc[item["id"]] = _candidate(mhc_graph)

    def test_decoder_shows_two_mhc_returns_and_decoder_residuals(self):
        for gid, candidate in self.decoders.items():
            ids = {comp["id"] for comp in candidate["components"]}
            for segment in ("attn", "ffn"):
                self.assertIn(f"n-m-{segment}_residual_combt", ids)
                self.assertIn(f"n-m-{segment}_residual_expand", ids)
                self.assertIn(f"n-m-{segment}_residual_mix", ids)
            mhc_node = "n-m-mhc_attn-0"
            self.assertIn(mhc_node, ids)
            labels = {
                str(conn.get("label") or "").split("｜")[0]: conn
                for conn in candidate["connections"]
                if conn["from"] == mhc_node
            }
            self.assertIn("collapsed", labels)
            self.assertIn("post", labels)
            self.assertIn("comb", labels)
            for key in ("collapsed", "post", "comb"):
                self.assertIn("｜batch_size", labels[key]["label"])
            # mHC 不能直接调用 Attention/MoE 节点。
            self.assertFalse(any(
                conn["from"] == mhc_node
                and conn["to"] in {
                    "n-m-attn_q-2", "n-m-attn_kv-3",
                    "n-m-attn_core-6", "n-m-attn_out-7",
                }
                for conn in candidate["connections"]
            ))
            # comb 必须先转置，再进入 combine + Add。
            self.assertTrue(any(
                conn["from"].startswith("n-m-attn_residual_combt")
                and conn["to"].startswith("n-m-attn_residual_mix")
                for conn in candidate["connections"]
            ))
            # 原始多流 + branch post 展开都进入 Add。
            mix_ids = {
                conn["id"] for conn in candidate["connections"]
                if conn["to"].startswith("n-m-attn_residual_mix")
            }
            self.assertTrue(mix_ids)

    def test_decoder_labels_carry_identity_and_full_shape(self):
        for gid, candidate in self.decoders.items():
            for conn in candidate["connections"]:
                text = str(conn.get("label") or "")
                if text.split("｜")[0] in (
                        "collapsed", "post", "comb", "branch_output",
                        "combᵀ", "原始多流", "混合后多流"):
                    self.assertIn("｜batch_size", text, (gid, text))

    def test_decoder_data_edges_flow_downward(self):
        for gid, candidate in self.decoders.items():
            pos = {
                comp["id"]: comp["pos"] for comp in candidate["components"]
            }
            for conn in candidate["connections"]:
                if conn.get("label") in ("collapsed", "post", "comb", "combᵀ"):
                    continue
                if conn["from"] in pos and conn["to"] in pos:
                    self.assertGreaterEqual(
                        pos[conn["to"]][1], pos[conn["from"]][1],
                        f"{gid}: {conn['id']} 向上回流",
                    )

    def test_mhc_detail_has_returns_but_no_branch_nodes(self):
        for gid, candidate in self.mhc.items():
            ids = {comp["id"] for comp in candidate["components"]}
            self.assertIn("n-o-consumer-attn", ids)
            self.assertIn("n-o-consumer-ffn", ids)
            self.assertNotIn("n-o-out", ids)
            labels = [str(comp.get("label", "")) for comp in candidate["components"]]
            self.assertFalse(any(
                label.startswith(("Attention —", "MoE 专家", "Hash 路由"))
                for label in labels
            ))
            returns = {
                str(conn.get("label") or "").split("｜")[0]
                for conn in candidate["connections"]
                if conn["to"] in ("n-o-consumer-attn", "n-o-consumer-ffn")
            }
            self.assertTrue({"collapsed", "post", "comb"} <= returns)
            for conn in candidate["connections"]:
                if conn["to"] in ("n-o-consumer-attn", "n-o-consumer-ffn"):
                    self.assertIn("｜batch_size", conn["label"])
            labels = " ".join(
                str(comp.get("label", "")) + " " + str(comp.get("sublabel", ""))
                for comp in candidate["components"]
            )
            self.assertIn("Sinkhorn", labels)
            self.assertIn("comb", labels)

    def test_mhc_detail_geometry_has_no_negative_or_clipped_outer_edge(self):
        for gid, candidate in self.mhc.items():
            components = candidate["components"]
            min_x = min(comp["pos"][0] for comp in components)
            min_y = min(comp["pos"][1] for comp in components)
            self.assertGreaterEqual(min_x, 0, f"{gid}: 节点负坐标")
            self.assertGreaterEqual(min_y, 0, f"{gid}: 节点负 y")
            for conn in candidate["connections"]:
                for point in conn.get("via") or []:
                    self.assertGreaterEqual(point[0], 0, f"{gid}: 路由负 x")
                    self.assertGreaterEqual(point[1], 0, f"{gid}: 路由负 y")
                at = conn.get("labelAt")
                if at:
                    width = layers._edge_label_width(conn.get("label") or "")
                    self.assertGreaterEqual(at[0] - width / 2, 0,
                                            f"{gid}: 标签左越界")
                    self.assertGreaterEqual(at[1] - 9, 0,
                                            f"{gid}: 标签上越界")

    def test_mhc_parameter_local_and_sinkhorn_loop(self):
        for gid, candidate in self.mhc.items():
            params = self.mhc_graphs[gid].parameter_items
            ids = {str(item.get("id")) for item in params}
            self.assertTrue({"Nhc", "Sit", "Eps"} <= ids, (gid, ids))
            self.assertTrue(ids <= {"B", "seqlen", "H", "Nhc", "Sit", "Eps"}, (gid, ids))
            comb = next(
                node for node in self.mhc_graphs[gid].nodes
                if "Sinkhorn" in str(node.label)
            )
            self.assertIn("1 次初始列归一", comb.sublabel)
            self.assertIn("(Sit-1)", comb.sublabel)
            details = " ".join(comb.details or ())
            self.assertIn("hc_sinkhorn_iters-1", details)
            self.assertIn("不复制 20 份节点", details)

    def test_mhc_detail_labels_do_not_cover_nodes(self):
        for gid, candidate in self.mhc.items():
            nodes = [
                (comp["pos"][0], comp["pos"][1],
                 comp["size"][0], comp["size"][1])
                for comp in candidate["components"]
            ]
            for conn in candidate["connections"]:
                at = conn.get("labelAt")
                text = conn.get("label")
                if not at or not text:
                    continue
                width = layers._edge_label_width(text)
                rect = (at[0] - width / 2, at[1] - 9, width, 10)
                for node in nodes:
                    self.assertFalse(
                        layers._rects_overlap(rect, node, 1.0),
                        f"{gid}: 标签 {text!r} 覆盖节点",
                    )


class TestD04Moe(unittest.TestCase):
    """D04：MoE 关系、Hash/TopK、激活专家、共享专家与汇合。"""

    @classmethod
    def setUpClass(cls):
        cls.items = _items()
        cls.relations = {}
        cls.routes = {}
        cls.experts = {}
        cls.shared = {}
        cls.combine = {}
        cls.mhc = {}
        for item in cls.items:
            gid = item["id"]; block = item["structure"]["block"]
            group = {"id": gid, "label": item.get("label", gid),
                     "representative_layer_index": item.get("representative_layer_index"),
                     "attention_type": item.get("attention_type"),
                     "moe_type": item.get("moe_type")}
            cls.relations[gid] = layers.build_moe_relation_graph(
                block, modules=layers.module_groups(block), group=group)
            cls.routes[gid] = layers.build_moe_route_graph(block, group=group)
            cls.experts[gid] = layers.build_moe_expert_graph(block, group=group)
            cls.shared[gid] = layers.build_moe_shared_graph(block, group=group)
            cls.combine[gid] = layers.build_moe_combine_graph(block, group=group)
            cls.mhc[gid] = layers.build_mhc_graph(
                block, modules=layers.module_groups(block), group=group)

    def test_relation_keeps_routing_and_parallel_shared(self):
        for gid, graph in self.relations.items():
            ids = {n.node_id for n in graph.nodes}
            self.assertTrue({"r-in", "r-route", "r-dispatch", "r-expert",
                             "r-shared", "r-combine", "r-out"} <= ids)
            pairs = {(e.src, e.dst) for e in graph.edges}
            self.assertIn(("r-in", "r-route"), pairs)
            self.assertIn(("r-in", "r-shared"), pairs)
            self.assertNotIn(("r-route", "r-shared"), pairs)
            self.assertIn(("r-route", "r-expert"), pairs)
            self.assertIn(("r-route", "r-dispatch"), pairs)
            self.assertIn(("r-shared", "r-combine"), pairs)
            self.assertIn(("r-expert", "r-combine"), pairs)
            self.assertIn(("r-dispatch", "r-combine"), pairs)
            self.assertIn(("r-combine", "r-out"), pairs)
            self.assertTrue(all(e.shape for e in graph.edges if e.kind == "data"), gid)
            combine_inputs = {(e.src, e.tensor_id) for e in graph.edges
                              if e.dst == "r-combine"}
            self.assertIn(("r-expert", "expert_down_out"), combine_inputs, gid)
            self.assertIn(("r-dispatch", "expert_token_idx"), combine_inputs, gid)
            self.assertNotIn(("r-expert", "routed_out"), combine_inputs, gid)

    def test_hash_and_topk_use_real_route_paths(self):
        for gid, graph in self.routes.items():
            op_ids = {op for n in graph.nodes for op in (n.op_ids or ())}
            node_ids = {n.node_id for n in graph.nodes}
            if gid in ("swa-hash", "csa-hash"):
                self.assertIn("moe_hash_lookup", op_ids)
                self.assertNotIn("m-router-bias", node_ids)
                self.assertFalse(any(e.tensor_id == "router_bias" for e in graph.edges))
            else:
                self.assertIn("moe_router", op_ids)
                self.assertIn("m-router-bias", node_ids)
                self.assertTrue(any(e.tensor_id == "router_bias" for e in graph.edges))
            self.assertFalse(any(e.tensor_id == "tid2eid" for e in graph.edges)
                             and gid not in ("swa-hash", "csa-hash"))
            router = next(n for n in graph.nodes if n.node_id == "m-router")
            text = f"{router.label} {router.sublabel}"
            self.assertIn("scores.gather", text, gid)
            self.assertIn("sum+1e-20", text, gid)
            self.assertIn("Scl", text, gid)
            self.assertIn("score_fn", text, gid)
            if gid in ("swa-hash", "csa-hash"):
                self.assertIn("tid2eid", text, gid)
                self.assertNotIn("e_score_correction_bias", text, gid)
            else:
                self.assertIn("e_score_correction_bias", text, gid)

    def test_single_expert_has_real_weight_slices_and_clamp(self):
        for gid, graph in self.experts.items():
            ids = {n.node_id for n in graph.nodes}
            self.assertTrue({"e-slice", "e-gu", "e-clamp", "e-act", "e-down", "e-weight"} <= ids)
            tensors = {e.tensor_id for e in graph.edges}
            self.assertIn("expert_gate_up_e", tensors)
            self.assertIn("expert_down_e", tensors)
            self.assertTrue(any(e.dst == "e-gu" and e.tensor_id == "expert_gate_up_e"
                                for e in graph.edges))
            self.assertTrue(any(e.dst == "e-down" and e.tensor_id == "expert_down_e"
                                for e in graph.edges))
            self.assertTrue(any(e.dst == "e-weight" and e.tensor_id == "topk_weights"
                                for e in graph.edges))
            self.assertTrue(any("expert_token_count" in (n.sublabel or "")
                                for n in graph.nodes) or
                            any("num_expert_tokens" in e.shape for e in graph.edges))

    def test_shared_expert_uses_full_tokens_and_combine_order(self):
        for gid, graph in self.shared.items():
            pairs = {(e.src, e.dst) for e in graph.edges}
            self.assertIn(("s-in", "sh-gate-up"), pairs)
            self.assertIn(("sh-gate-up", "sh-act-down"), pairs)
            self.assertTrue(any(e.tensor_id == "ffn_normed" for e in graph.edges))
            self.assertFalse(any(e.tensor_id == "expert_tokens" for e in graph.edges))
            shared_params = {str(p.get("id")) for p in graph.parameter_items}
            self.assertIn("Lim", shared_params, gid)
            self.assertNotIn("Eps", shared_params, gid)
            gate_up = next(n for n in graph.nodes if n.node_id == "sh-gate-up")
            act_down = next(n for n in graph.nodes if n.node_id == "sh-act-down")
            self.assertIn("Clamp", gate_up.label, gid)
            self.assertIn("SiLU", act_down.label, gid)
        for gid, graph in self.combine.items():
            pairs = {(e.src, e.dst) for e in graph.edges}
            self.assertIn(("s-in", "c-add"), pairs)
            self.assertIn(("c-add", "c-view"), pairs)
            self.assertIn(("s-in", "c-merge"), pairs)
            self.assertIn(("c-view", "c-merge"), pairs)
            self.assertTrue(any(e.dst == "s-out" and e.tensor_id == "mlp_output"
                                for e in graph.edges))
            add = next(n for n in graph.nodes if n.node_id == "c-add")
            self.assertIn("moe_combine", add.op_ids, gid)
            self.assertIn("index_add_", add.sublabel, gid)
            add_inputs = {(e.src, e.tensor_id) for e in graph.edges if e.dst == "c-add"}
            self.assertIn(("s-in", "expert_down_out"), add_inputs, gid)
            self.assertIn(("s-in", "expert_token_idx"), add_inputs, gid)
            self.assertFalse(any(e.dst == "c-add" and e.tensor_id == "topk_weights"
                                 for e in graph.edges), gid)

    def test_moe_edge_main_text_keeps_identity_and_full_shape(self):
        for graph in list(self.relations.values()) + list(self.routes.values()) +                 list(self.experts.values()) + list(self.shared.values()) + list(self.combine.values()):
            for edge in graph.edges:
                if edge.kind != "data":
                    continue
                text = layers._edge_display_text(edge, graph)
                self.assertIn(edge.tensor_label, text, graph.key)
                self.assertIn(edge.shape, text, graph.key)
                self.assertIn("｜", text, graph.key)

    def test_mhc_card_carries_local_source_range(self):
        for gid, graph in self.mhc.items():
            cards = layers._level_cards(graph, None, None)
            items = [item for card in cards for item in card.get("items", [])]
            self.assertTrue(any("modeling 926-938" in item for item in items), gid)
            self.assertTrue(any("hc_sinkhorn_iters" in item for item in items), gid)


class TestD03AttentionCache(unittest.TestCase):
    """D03：Attention/Q·KV·O、三类缓存、压缩器与 Indexer。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_semantic import build_semantic_bundle
        from structure_reader.render.archify_preview import (
            build_candidate,
            build_view_model,
        )

        cls.items = _items()
        cls.semantic = {}
        cls.bundles = {}
        cls.cache_graphs = {}
        cls.compressor_graphs = {}
        cls.indexer_graphs = {}
        cls.q_graphs = {}
        cls.o_graphs = {}
        for item in cls.items:
            gid = item["id"]
            block = item["structure"]["block"]
            modules = layers.module_groups(block)
            group = {
                "id": gid,
                "label": item.get("label", gid),
                "representative_layer_index": item.get("representative_layer_index"),
                "attention_type": item.get("attention_type"),
            }
            bundle = build_semantic_bundle(
                REPO_ROOT, layer_index=item["representative_layer_index"])
            attention_view = build_view_model(bundle, "attention")
            cls.bundles[gid] = bundle
            cls.semantic[gid] = (
                attention_view,
                build_candidate(
                    bundle, attention_view, output_relpath=f"x/{gid}-attention.html",
                    form="normalized", normalization_report=None,
                    tool={"name": "Archify", "version": "3.2.0", "sha256": "x"},
                    repository=REPOSITORY,
                ),
            )
            parts = ["main"]
            if item.get("attention_type") != "sliding_attention":
                parts.append("compressor")
            if item.get("attention_type") == "compressed_sparse_attention":
                parts.append("indexer")
            cls.cache_graphs[gid] = {
                part: layers.build_cache_graph(block, group=group, part=part)
                for part in parts
            }
            cls.q_graphs[gid] = layers.build_q_graph(
                block, modules=modules, group=group)
            cls.o_graphs[gid] = layers.build_o_graph(
                block, modules=modules, group=group)
            if item.get("attention_type") != "sliding_attention":
                cls.compressor_graphs[gid] = layers.build_compressor_graph(
                    block, modules=modules, group=group)
            if item.get("attention_type") == "compressed_sparse_attention":
                cls.indexer_graphs[gid] = layers.build_indexer_graph(
                    block, modules=modules, group=group)

    def test_attention_overview_keeps_parallel_qkv_and_block_bias(self):
        for gid, (view, candidate) in self.semantic.items():
            ids = {node.semantic_id for node in view.nodes}
            bundle = self.bundles[gid]
            edge_pairs = {(edge.src_node_id, edge.dst_node_id,
                           edge.tensor_id) for edge in bundle.graph.edges}
            self.assertFalse(any(
                src.startswith("attn.q") and dst.startswith("attn.kv")
                for src, dst, _ in edge_pairs
            ), f"{gid}: 伪造 Q→KV 顺序依赖")
            if "attn.block_bias" in ids:
                self.assertTrue(any(
                    src == "attn.block_bias" and dst == "attn.mask_extend"
                    for src, dst, _ in edge_pairs
                ))
                # 最终 candidate 的实际连接，不只 bundle 字段。
                final_edges = {
                    (c["from"], c["to"]): c.get("label")
                    for c in candidate["connections"]
                }
                self.assertTrue(any(
                    src.startswith("n-attn-compressor")
                    and dst.startswith("n-attn-block_bias")
                    for src, dst in final_edges
                ), f"{gid}: candidate 丢 compressor→block_bias")
                self.assertTrue(any(
                    src.startswith("n-attn-block_bias")
                    and dst.startswith("n-attn-mask_extend")
                    for src, dst in final_edges
                ), f"{gid}: candidate 丢 block_bias→mask_extend")
                if "attn.indexer" in ids:
                    self.assertTrue(any(
                        src.startswith("n-attn-indexer")
                        and dst.startswith("n-attn-block_bias")
                        for src, dst in final_edges
                    ), f"{gid}: candidate 丢 Indexer→block_bias")
                    self.assertTrue(any(
                        src.startswith("n-attn-block_bias")
                        and "num_compressed" in str(label)
                        for (src, _dst), label in final_edges.items()
                    ), f"{gid}: candidate 缺 block_bias shape")
                else:
                    self.assertTrue(any(
                        "position" in src and "block_bias" in dst
                        for src, dst in final_edges
                    ), f"{gid}: HCA candidate 缺 position_ids→block_bias")
            self.assertFalse(any(
                src == "attn.indexer" and dst == "attn.compressor"
                for src, dst, _ in edge_pairs
            ), f"{gid}: topk 不应作为压缩器数据输入")
            if "attn.indexer" in ids:
                self.assertTrue(any(
                    src == "attn.q_a_norm" and dst == "attn.indexer"
                    for src, dst, _ in edge_pairs
                ), f"{gid}: Indexer 应接收 q_residual")

    def test_hca_block_bias_is_not_csa(self):
        hca = self.bundles["hca-topk"].nodes["attn.block_bias"]
        self.assertIn("HCA", hca.op_type)
        self.assertNotIn("CSA", hca.op_type)
        self.assertNotIn("index_topk", hca.evidence_refs)
        self.assertIn("heavily_compressed_attention",
                      self.bundles["hca-topk"].node_meta["attn.block_bias"].condition)
        csa = self.bundles["csa-hash"].nodes["attn.block_bias"]
        self.assertIn("CSA", csa.op_type)
        self.assertIn("compressed_sparse_attention",
                      self.bundles["csa-hash"].node_meta["attn.block_bias"].condition)

    def test_cache_graphs_split_swa_csa_hca(self):
        for gid, parts in self.cache_graphs.items():
            main = parts["main"]
            ids = {node.node_id for node in main.nodes}
            self.assertIn("cache-main-in", ids)
            self.assertIn("cache-main-new-kv", ids)
            self.assertIn("cache-main-full", ids)
            self.assertIn("cache-main-out", ids)
            self.assertTrue(any(node.op_ids == ("cache_update",) for node in main.nodes))
            if gid == "swa-hash":
                self.assertEqual(set(parts), {"main"})
            elif gid == "hca-topk":
                self.assertEqual(set(parts), {"main", "compressor"})
            else:
                self.assertEqual(set(parts), {"main", "compressor", "indexer"})
            for part, graph in parts.items():
                for edge in graph.edges:
                    if edge.kind == "data":
                        self.assertTrue(edge.shape, (gid, part, edge.src, edge.dst))

    def test_cache_has_every_tensor_identity_and_shape(self):
        for gid, parts in self.cache_graphs.items():
            seen = set()
            for graph in parts.values():
                for edge in graph.edges:
                    if edge.members:
                        for member in edge.members:
                            seen.add((member["tensor_id"], member["shape"]))
                for node in graph.nodes:
                    for detail in node.details or ():
                        if "tensor" in detail:
                            seen.add((detail, ""))
            block = next(item for item in self.items if item["id"] == gid)["structure"]["block"]
            expected = {
                str(t["id"])
                for t in block.get("tensors", []) if t.get("kind") == "cache"
                and not str(t["id"]).startswith(("tid2eid", "kv_full"))
            }
            seen_ids = {tid for tid, _shape in seen}
            self.assertEqual(expected - seen_ids, set(), f"{gid}: 缓存逐项未全部可见")
            self.assertTrue(all(shape for tid, shape in seen if tid in expected),
                            f"{gid}: 缓存逐项缺完整 shape")

    def test_compressor_and_indexer_keep_named_ops(self):
        for gid, graph in self.compressor_graphs.items():
            self.assertTrue(any(node.op_ids and any(
                op.startswith(("c_", "hca_")) for op in node.op_ids)
                for node in graph.nodes), gid)
            if gid == "hca-topk":
                self.assertTrue(any("hca_block_bias" in (node.op_ids or ())
                                    for node in graph.nodes), gid)
                self.assertTrue(any("c_mask_extend" in (node.op_ids or ())
                                    for node in graph.nodes), gid)
        for gid, graph in self.indexer_graphs.items():
            self.assertTrue(any(node.op_ids and node.op_ids[0].startswith("i_")
                                for node in graph.nodes), gid)
            ids = {node.node_id for node in graph.nodes}
            self.assertFalse(any("compressed_c" in node for node in ids),
                             "Indexer 细图不应把外层压缩 KV 当成唯一输入")

    def test_q_and_o_have_independent_graphs(self):
        for gid, qgraph in self.q_graphs.items():
            self.assertTrue(any(node.op_ids and "q_rope_apply" in node.op_ids
                                for node in qgraph.nodes), gid)
            self.assertTrue(any("q_b_norm" in " ".join(node.op_ids or ())
                                for node in qgraph.nodes), gid)
            ograph = self.o_graphs[gid]
            self.assertTrue(any("o_a_proj" in " ".join(node.op_ids or ())
                                for node in ograph.nodes), gid)
            self.assertTrue(any("o_b_proj" in " ".join(node.op_ids or ())
                                for node in ograph.nodes), gid)

    def test_stage_graphs_cover_all_real_op_inputs(self):
        for gid in self.compressor_graphs:
            for name, graph, prefix in (
                ("compressor", self.compressor_graphs[gid], ("c_", "hca_", "mask_")),
            ):
                incoming = {(edge.dst, edge.tensor_id) for edge in graph.edges}
                for node in graph.nodes:
                    for op_id in node.op_ids or ():
                        for tid in self._op_inputs(gid, op_id):
                            if tid in self._weights(gid) or tid in self._stage_outputs(gid, node):
                                continue
                            self.assertIn((node.node_id, tid), incoming,
                                          f"{gid}:{name}:{node.node_id} 缺真实输入 {tid}")
        for gid, graph in self.indexer_graphs.items():
            incoming = {(edge.dst, edge.tensor_id) for edge in graph.edges}
            for node in graph.nodes:
                for op_id in node.op_ids or ():
                    for tid in self._op_inputs(gid, op_id):
                        if tid in self._weights(gid) or tid in self._stage_outputs(gid, node):
                            continue
                        self.assertIn((node.node_id, tid), incoming,
                                      f"{gid}:indexer:{node.node_id} 缺真实输入 {tid}")
            score_inputs = self._op_inputs(gid, "i_scorer")
            score_edges = {edge.tensor_id for edge in graph.edges if edge.dst == "i-score"}
            self.assertTrue(set(score_inputs) <= score_edges | self._weights(gid),
                            f"{gid}: scorer 输入遗漏 {set(score_inputs) - score_edges}")

    def _stage_outputs(self, gid, node):
        block = next(i for i in self.items if i["id"] == gid)["structure"]["block"]
        ops = {o["id"]: o for o in block["operators"]}
        return {str(t) for op in (node.op_ids or ()) for t in ops[op].get("outputs", [])}

    def _op_inputs(self, gid, op_id):
        block = next(i for i in self.items if i["id"] == gid)["structure"]["block"]
        op = next(o for o in block["operators"] if o["id"] == op_id)
        return [str(t) for t in op.get("inputs", [])]

    def _weights(self, gid):
        block = next(i for i in self.items if i["id"] == gid)["structure"]["block"]
        return {str(t["id"]) for t in block.get("tensors", []) if t.get("kind") == "weight"}

    def test_stage_external_fanout_keeps_internal_and_boundary_edges(self):
        for gid in ("csa-hash", "csa-topk"):
            graph = self.q_graphs[gid]
            internal = [e for e in graph.edges
                        if e.src == "q-lowrank" and e.dst == "q-b"
                        and e.tensor_id == "q_a_normed"]
            boundary = [e for e in graph.edges
                        if e.dst == "s-out" and e.tensor_id == "q_a_normed"]
            self.assertTrue(internal, f"{gid}: q_a_normed 内部边丢失")
            self.assertTrue(boundary, f"{gid}: q_a_normed 图外出口丢失")
            self.assertIn("i_q_proj", boundary[0].condition or "")
        for gid in ("csa-hash", "csa-topk", "hca-topk"):
            graph = self.compressor_graphs[gid]
            internal = [e for e in graph.edges
                        if e.src == "s-state" and e.dst == "s-bias"
                        and e.tensor_id == "cache_compressed_c_post"]
            boundary = [e for e in graph.edges
                        if e.dst == "s-out" and e.tensor_id == "cache_compressed_c_post"]
            self.assertTrue(internal, f"{gid}: compressed 内部 bias 边丢失")
            self.assertTrue(boundary, f"{gid}: compressed 图外 kv_concat 出口丢失")
            self.assertIn("kv_concat", boundary[0].condition or "")
            # 纯内部/普通中间张量不冒充段输出。
            self.assertFalse(any(e.dst == "s-out" and e.tensor_id in
                                 ("c_chunk_kv", "c_windows_kv", "c_positions", "first_window_position")
                                 for e in graph.edges), f"{gid}: 内部张量被泄漏为段输出")

    def test_stage_graph_rejects_unknown_ids(self):
        import copy
        from structure_reader.render.archify_layers import ArchifyLayersError
        block = copy.deepcopy(next(i for i in self.items if i["id"] == "csa-hash")["structure"]["block"])
        with self.assertRaises(ArchifyLayersError):
            layers._stage_graph(
                block, stages=(("s-x", "x", ("no_such_op",)),),
                key="x", level="compressor", title="x", subtitle="x",
                entry_tensor="attn_normed", variant="csa-hash", group="csa-hash")
        bad = copy.deepcopy(block)
        op = next(o for o in bad["operators"] if o["id"] == "c_kv_gate_proj")
        op["inputs"] = list(op.get("inputs", [])) + ["no_such_tensor"]
        with self.assertRaises(ArchifyLayersError):
            layers._stage_graph(
                bad, stages=(("s-proj", "x", ("c_kv_gate_proj",)),),
                key="x", level="compressor", title="x", subtitle="x",
                entry_tensor="attn_normed", variant="csa-hash", group="csa-hash")

    def test_csa_bias_rank_and_cache_cumlen_projection(self):
        csa = self.compressor_graphs["csa-hash"]
        mask_edges = [e for e in csa.edges if e.dst == "s-mask"]
        bias_edges = [e for e in mask_edges if "c_block_bias" in (e.tensor_id or "")]
        self.assertTrue(bias_edges, "CSA mask 缺 c_block_bias 真实输入")
        self.assertNotEqual(bias_edges[0].shape, "标量")
        self.assertIn("num_compressed", bias_edges[0].shape)
        main = self.cache_graphs["csa-hash"]["main"]
        self.assertTrue(any(n.node_id == "cache-main-cumlen-post" and
                            "kvcachelen+seqlen" in n.sublabel for n in main.nodes))

    def test_cache_table_uses_full_symbols_and_projection(self):
        from structure_reader.render.hf_reading_page import _cache_state_section
        block = next(i for i in self.items if i["id"] == "csa-hash")["structure"]["block"]
        html = _cache_state_section(block, "csa-hash")
        self.assertIn("cache_cumlen_post（阅读投影）", html)
        self.assertIn("kvcachelen+seqlen", html)
        self.assertIn("batch_size", html)
        self.assertNotIn("B, 1, min", html)
        self.assertIn("真实读写来源", html)
        self.assertNotIn("写: —", html)

    def test_d03_graphs_have_nonnegative_geometry(self):
        graphs = {}
        for gid, parts in self.cache_graphs.items():
            for part, graph in parts.items():
                graphs[f"{gid}-{part}"] = graph
        graphs.update(self.compressor_graphs)
        graphs.update(self.indexer_graphs)
        graphs.update(self.q_graphs)
        graphs.update(self.o_graphs)
        for gid, graph in graphs.items():
            candidate = _candidate(graph)
            for comp in candidate["components"]:
                self.assertGreaterEqual(comp["pos"][0], 0, gid)
                self.assertGreaterEqual(comp["pos"][1], 0, gid)
            for conn in candidate["connections"]:
                for point in conn.get("via") or []:
                    self.assertGreaterEqual(point[0], 0, gid)
                    self.assertGreaterEqual(point[1], 0, gid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
