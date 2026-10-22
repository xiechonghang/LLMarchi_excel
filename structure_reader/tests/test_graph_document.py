"""S1 01a SemanticGraph 与 canonical JSON 独立验证。

范围严格限定于 `dispatch/development_01a_graph_document_probe.md` 的契约、
`results/development_01a_graph_document_probe.md` 的交付说明，以及
`dispatch/repair_01a_graph_document_boundaries.md` 的返修要求：

- `SemanticGraph`（frozen）：显式 `schema`（`semantic-graph/v1`）、`template_id`、
  `world_size`（**固定 1 的分析假设**）+ 十个有序实体集合
  （`named_axes`/`tensors`/`nodes`/`edges`/`state_versions`/`state_transitions`/
  `guards`/`guard_activations`/`alias_relations`/`fold_groups`；Port 依附 Node）。
- `graph_to_json`：实体集合按 `stable_id` 升序、`GuardActivation` 按
  `(guard_id, scenario_id)` 组合键；**只排序实体集合**，张量轴/布局、节点端口、
  证据、折叠映射等语义内部顺序**不重排**；同一图不同录入顺序产出**逐字节相同** JSON。
- `graph_from_json`：非字符串/非法 JSON/顶层非对象/`schema` 不符受控拒绝；
  `world_size` 与**十个必需集合全部显式存在**，缺失全部点名；集合必须是 JSON 数组；
  实体构造错误定位 `集合[索引]`；**反序列化阶段**即拒绝重复 `stable_id` 与
  重复 `(guard_id, scenario_id)` 组合键（报集合、第二次索引与重复键）；不抛裸
  `KeyError`/`TypeError`/`AttributeError`。
- 返修点：缺失集合/`world_size` 具名拒绝；`graph_from_json(None)` 与错类实体为受控
  `ValueError`；重复键在反序列化阶段即拒绝；组合键用**真实二元 tuple**（不用拼串），
  在含 `\u0000` 的字符串下不发生拼接碰撞。

**明确属图级语义核验、本单不通过也不声称**（见 `GraphLevelSemanticsGapsTest`）：
端点/轴/版本/alias/guard 引用一致性、端口方向匹配、无版本环、折叠连通性等价。
**文档可序列化不等于图正确。**

不在本范围：真实 DeepSeek 结构提取、template/instance 实例绑定、完整来源证据对象、
01b 检查器、01c schema、求值器（开发单 02）、源码核验（开发单 03）、阅读页/CSV/Skill。

纪律：数据为**合成契约样例（synthetic contract fixture）**，非 DeepSeek 结构；期望值来自
任务单与开发/返修报告声明的接口，**不复制产品当前输出为 golden**；只用标准库；不安装/导入
jsonschema、torch、transformers、safetensors。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import structure_reader.graph_ir as gir  # noqa: E402
import structure_reader.graph_document as gdoc  # noqa: E402

SCHEMA = "semantic-graph/v1"
COLLECTIONS = ("named_axes", "tensors", "nodes", "edges", "state_versions",
               "state_transitions", "guards", "guard_activations",
               "alias_relations", "fold_groups")
ENVELOPE_KEYS = {"schema", "template_id", "world_size"} | set(COLLECTIONS)
TEMPLATE_ID = "template:csa_block"


# ---------------------------------------------------------------------------
# 合成契约样例（非 DeepSeek 结构）
# ---------------------------------------------------------------------------
def axis_B():
    return gir.NamedAxis(symbol="B", name="batch", meaning="批大小", origin="fixture:axes",
                         evidence_ref="fixture:axes/B", value=1)


def axis_H():
    return gir.NamedAxis(symbol="H", name="hidden", meaning="隐藏维", origin="fixture:axes",
                         evidence_ref="fixture:axes/H", value=4096)


def tensor_hidden(axes=None, logical=None, physical=None):
    axes = [axis_B(), axis_H()] if axes is None else axes
    symbols = [a.symbol for a in axes]
    return gir.Tensor(name="hidden", dtype="bf16", axes=axes,
                      logical_layout=symbols if logical is None else logical,
                      physical_layout=symbols if physical is None else physical)


def tensor_kv_cache():
    return gir.Tensor(name="kv_cache", dtype="bf16", axes=[axis_B(), axis_H()],
                      logical_layout=["B", "H"], physical_layout=["H", "B"])


def node_proj(**overrides):
    spec = {
        "stable_id": "call:proj@layer0", "op_type": "proj", "granularity": "op",
        "input_ports": [gir.Port(name="proj_in", direction="input",
                                 tensor_id="tensor:hidden")],
        "output_ports": [gir.Port(name="proj_out", direction="output",
                                  tensor_id="tensor:hidden")],
        "evidence_refs": ["fixture:node/proj"],
    }
    spec.update(overrides)
    return gir.Node(**spec)


def node_attn(**overrides):
    spec = {
        "stable_id": "call:attn@layer0", "op_type": "attn", "granularity": "op",
        "input_ports": [
            gir.Port(name="attn_q", direction="input", tensor_id="tensor:hidden"),
            gir.Port(name="attn_kv", direction="input", tensor_id="tensor:kv_cache"),
        ],
        "output_ports": [gir.Port(name="attn_out", direction="output",
                                  tensor_id="tensor:hidden")],
        "evidence_refs": ["fixture:node/attn"],
    }
    spec.update(overrides)
    return gir.Node(**spec)


def edge_data(**overrides):
    spec = {
        "stable_id": "edge:data:proj->attn",
        "src_node_id": "call:proj@layer0", "src_port": "proj_out",
        "dst_node_id": "call:attn@layer0", "dst_port": "attn_q",
        "tensor_id": "tensor:hidden", "kind": "data",
        "evidence_refs": ["fixture:edge/data"],
    }
    spec.update(overrides)
    return gir.Edge(**spec)


def edge_state(**overrides):
    spec = {
        "stable_id": "edge:state:kv",
        "src_node_id": "call:attn@layer0", "src_port": "attn_kv",
        "dst_node_id": "call:attn@layer0", "dst_port": "attn_kv",
        "tensor_id": "tensor:kv_cache", "kind": "state",
        "evidence_refs": ["fixture:edge/state"],
        "state_transition_ref": "statetrans:kv:t0->t1",
    }
    spec.update(overrides)
    return gir.Edge(**spec)


def state_versions():
    return [gir.StateVersion(stable_id="statever:kv@t0", state_id="state:kv",
                             version_label="t0"),
            gir.StateVersion(stable_id="statever:kv@t1", state_id="state:kv",
                             version_label="t1")]


def state_transition(**overrides):
    spec = {"stable_id": "statetrans:kv:t0->t1", "state_id": "state:kv",
            "before_version_id": "statever:kv@t0", "after_version_id": "statever:kv@t1",
            "evidence_refs": ["fixture:state/trans"]}
    spec.update(overrides)
    return gir.StateTransition(**spec)


def guard_causal(**overrides):
    spec = {"stable_id": "guard:causal", "predicate_expr": "q_idx >= k_idx",
            "evidence_refs": ["fixture:guard/causal"]}
    spec.update(overrides)
    return gir.Guard(**spec)


def activation(**overrides):
    spec = {"guard_id": "guard:causal", "scenario_id": "scenario:prefill",
            "active_for_scenario": "true"}
    spec.update(overrides)
    return gir.GuardActivation(**spec)


def alias_key(**overrides):
    spec = {"stable_id": "alias:kv#key", "base_tensor_id": "tensor:kv_cache",
            "alias_tensor_id": "tensor:kv_cache", "relation_kind": "same_tensor",
            "alias_role": "key", "evidence_refs": ["fixture:alias/key"]}
    spec.update(overrides)
    return gir.AliasRelation(**spec)


def boundary_mappings():
    """任务单要求的三类边界：data 输入、state 输出、select 旁路（按此序）。"""
    return [
        gir.FoldBoundaryMapping(direction="input", external_port="hidden_states",
                                edge_kind="data", internal_node_id="call:proj@layer0",
                                internal_port="proj_in",
                                evidence_refs=["fixture:fold/in"]),
        gir.FoldBoundaryMapping(direction="output", external_port="kv_state",
                                edge_kind="state", internal_node_id="call:attn@layer0",
                                internal_port="attn_kv",
                                evidence_refs=["fixture:fold/state_a",
                                               "fixture:fold/state_b"]),
        gir.FoldBoundaryMapping(direction="input", external_port="route_hint",
                                edge_kind="select", internal_node_id="call:attn@layer0",
                                internal_port="attn_q",
                                evidence_refs=["fixture:fold/sel"]),
    ]


def fold_group(**overrides):
    spec = {"stable_id": "fold:csa_block@layer0",
            "member_node_ids": ["call:proj@layer0", "call:attn@layer0"],
            "boundary_mappings": boundary_mappings()}
    spec.update(overrides)
    return gir.FoldGroup(**spec)


def graph_kwargs(named_axes=None, tensors=None, nodes=None, edges=None,
                 state_versions_=None, state_transitions=None, guards=None,
                 guard_activations=None, alias_relations=None, fold_groups=None,
                 **overrides):
    spec = {
        "template_id": TEMPLATE_ID,
        "named_axes": [axis_B(), axis_H()] if named_axes is None else named_axes,
        "tensors": [tensor_hidden(), tensor_kv_cache()] if tensors is None else tensors,
        "nodes": [node_proj(), node_attn()] if nodes is None else nodes,
        "edges": [edge_data(), edge_state()] if edges is None else edges,
        "state_versions": state_versions() if state_versions_ is None else state_versions_,
        "state_transitions": [state_transition()] if state_transitions is None else state_transitions,
        "guards": [guard_causal()] if guards is None else guards,
        "guard_activations": [activation()] if guard_activations is None else guard_activations,
        "alias_relations": [alias_key()] if alias_relations is None else alias_relations,
        "fold_groups": [fold_group()] if fold_groups is None else fold_groups,
    }
    spec.update(overrides)
    return spec


def full_graph(**overrides):
    return gdoc.SemanticGraph(**graph_kwargs(**overrides))


def full_json(**overrides):
    return gdoc.graph_to_json(full_graph(**overrides))


def payload_of(text):
    return json.loads(text)


class GraphDocumentTestCase(unittest.TestCase):
    def assertValueErrorLocated(self, callable_, *needles):
        with self.assertRaises(ValueError) as ctx:
            callable_()
        message = str(ctx.exception)
        if needles:
            self.assertTrue(
                any(needle in message for needle in needles),
                f"错误信息未定位到 {needles!r}；实际信息：{message!r}",
            )
        return ctx.exception

    def assert_no_bare_exception(self, callable_):
        """受控失败：要么成功，要么 ValueError；不得裸 KeyError/TypeError/AttributeError。"""
        try:
            callable_()
        except ValueError:
            return "ValueError"
        except (KeyError, TypeError, AttributeError) as exc:  # pragma: no cover
            self.fail(f"出现裸 {type(exc).__name__}: {exc}")
        return "ok"


# ---------------------------------------------------------------------------
# 1. 接口与 envelope：schema、template_id、world_size、十个集合
# ---------------------------------------------------------------------------
class GraphEnvelopeTest(GraphDocumentTestCase):
    def test_semantic_graph_is_frozen_dataclass_with_expected_fields(self):
        self.assertTrue(gdoc.SemanticGraph.__dataclass_params__.frozen)
        names = tuple(f.name for f in fields(gdoc.SemanticGraph))
        for required in ("template_id",) + COLLECTIONS + ("schema", "world_size"):
            with self.subTest(field=required):
                self.assertIn(required, names)

    def test_envelope_has_exactly_schema_template_and_ten_collections(self):
        payload = payload_of(full_json())
        self.assertEqual(set(payload), ENVELOPE_KEYS)
        self.assertEqual(len(payload), 13)

    def test_schema_is_recorded_and_fixed(self):
        self.assertEqual(payload_of(full_json())["schema"], SCHEMA)
        self.assertEqual(gdoc.SemanticGraph(template_id="probe").schema, SCHEMA)
        for bad in ("semantic-graph/v2", "", None, 1):
            with self.subTest(schema=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: gdoc.SemanticGraph(template_id="probe", schema=b), "schema")

    def test_world_size_one_recorded_explicitly(self):
        payload = payload_of(full_json())
        self.assertIn("world_size", payload)
        self.assertEqual(payload["world_size"], 1)
        self.assertEqual(full_graph().world_size, 1)

    def test_world_size_other_values_rejected_at_construction(self):
        for bad in (0, 2, 8, -1, 1.0, "1", None, True):
            with self.subTest(world_size=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: gdoc.SemanticGraph(template_id="probe", world_size=b),
                    "world_size")

    def test_world_size_other_values_rejected_at_from_json(self):
        for bad in (0, 2, 8, -1, "1", None, True):
            payload = payload_of(full_json())
            payload["world_size"] = bad
            with self.subTest(world_size=bad):
                self.assertValueErrorLocated(
                    lambda p=payload: gdoc.graph_from_json(json.dumps(p)), "world_size")

    def test_template_id_rules(self):
        for bad in ("", "   ", None, 5, True):
            with self.subTest(template_id=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: gdoc.SemanticGraph(template_id=b), "template_id")
        self.assertValueErrorLocated(
            lambda: gdoc.SemanticGraph(template_id=" t "), "template_id")

    def test_all_collections_are_sequences_with_empty_defaults(self):
        graph = gdoc.SemanticGraph(template_id="probe")
        for collection in COLLECTIONS:
            with self.subTest(collection=collection):
                self.assertEqual(getattr(graph, collection), ())

    def test_empty_graph_serializes_with_all_arrays_present(self):
        payload = payload_of(gdoc.graph_to_json(gdoc.SemanticGraph(template_id="probe")))
        for collection in COLLECTIONS:
            with self.subTest(collection=collection):
                self.assertEqual(payload[collection], [])
        self.assertEqual(payload["world_size"], 1)

    def test_collections_frozen_to_tuples_with_defensive_copy(self):
        nodes = [node_proj(), node_attn()]
        graph = gdoc.SemanticGraph(**graph_kwargs(nodes=nodes))
        self.assertIsInstance(graph.nodes, tuple)
        nodes.append(node_proj(stable_id="call:extra@layer0"))
        self.assertEqual(len(graph.nodes), 2)

    def test_frozen_prevents_mutation(self):
        graph = full_graph()
        with self.assertRaises(FrozenInstanceError):
            graph.template_id = "other"
        with self.assertRaises(FrozenInstanceError):
            graph.nodes = ()


# ---------------------------------------------------------------------------
# 2. canonical 字节稳定：集合录入顺序无关
# ---------------------------------------------------------------------------
class CanonicalOrderingTest(GraphDocumentTestCase):
    def _reversed(self, **overrides):
        kwargs = graph_kwargs(**overrides)
        for collection in COLLECTIONS:
            kwargs[collection] = list(reversed(kwargs[collection]))
        return gdoc.SemanticGraph(**kwargs)

    def test_all_collections_reversed_gives_identical_bytes(self):
        baseline = full_json()
        self.assertEqual(gdoc.graph_to_json(self._reversed()), baseline)

    def test_each_collection_independently_reordered_gives_identical_bytes(self):
        baseline = full_json()
        for collection in COLLECTIONS:
            kwargs = graph_kwargs()
            kwargs[collection] = list(reversed(kwargs[collection]))
            with self.subTest(collection=collection):
                self.assertEqual(gdoc.graph_to_json(gdoc.SemanticGraph(**kwargs)), baseline)

    def test_rotated_orders_give_identical_bytes(self):
        baseline = full_json()
        for shift in range(3):
            kwargs = graph_kwargs()
            for collection in COLLECTIONS:
                items = kwargs[collection]
                kwargs[collection] = items[shift:] + items[:shift]
            with self.subTest(shift=shift):
                self.assertEqual(gdoc.graph_to_json(gdoc.SemanticGraph(**kwargs)), baseline)

    def test_canonical_json_is_ascii_unescaped_and_compact(self):
        text = full_json()
        self.assertNotIn("\\u", text)
        self.assertNotIn(", ", text)
        self.assertNotIn(": ", text)
        self.assertIn("批大小", text)

    def test_collection_order_in_json_is_ascending_stable_id(self):
        payload = payload_of(full_json())
        self.assertEqual([a["symbol"] for a in payload["named_axes"]], ["B", "H"])
        for collection, key in (("nodes", "stable_id"), ("tensors", "stable_id"),
                                ("edges", "stable_id"), ("guards", "stable_id"),
                                ("fold_groups", "stable_id"),
                                ("alias_relations", "stable_id"),
                                ("state_versions", "stable_id"),
                                ("state_transitions", "stable_id")):
            with self.subTest(collection=collection):
                ids = [item[key] for item in payload[collection]]
                self.assertEqual(ids, sorted(ids))

    def test_guard_activations_sorted_by_composite_key(self):
        activations = [
            activation(guard_id="guard:b", scenario_id="scenario:1"),
            activation(guard_id="guard:a", scenario_id="scenario:2"),
            activation(guard_id="guard:a", scenario_id="scenario:1"),
        ]
        payload = payload_of(full_json(guard_activations=activations))
        pairs = [(a["guard_id"], a["scenario_id"]) for a in payload["guard_activations"]]
        self.assertEqual(pairs, sorted(pairs))


# ---------------------------------------------------------------------------
# 3. 语义内部顺序不重排
# ---------------------------------------------------------------------------
class InternalOrderPreservedTest(GraphDocumentTestCase):
    def test_tensor_axes_and_layouts_keep_declared_order(self):
        tensor = tensor_hidden(axes=[axis_H(), axis_B()], logical=["H", "B"],
                               physical=["B", "H"])
        payload = payload_of(full_json(tensors=[tensor]))
        entry = payload["tensors"][0]
        self.assertEqual([a["symbol"] for a in entry["axes"]], ["H", "B"])
        self.assertEqual(entry["logical_layout"], ["H", "B"])
        self.assertEqual(entry["physical_layout"], ["B", "H"])

    def test_tensor_axis_order_survives_reversed_collection_order(self):
        tensor = tensor_hidden(axes=[axis_H(), axis_B()], logical=["H", "B"],
                               physical=["B", "H"])
        other = tensor_kv_cache()
        forward = payload_of(full_json(tensors=[tensor, other]))
        backward = payload_of(full_json(tensors=[other, tensor]))
        for payload in (forward, backward):
            entry = [t for t in payload["tensors"] if t["name"] == "hidden"][0]
            self.assertEqual([a["symbol"] for a in entry["axes"]], ["H", "B"])
            self.assertEqual(entry["logical_layout"], ["H", "B"])

    def test_node_port_order_preserved(self):
        payload = payload_of(full_json())
        attn = [n for n in payload["nodes"] if n["stable_id"] == "call:attn@layer0"][0]
        self.assertEqual([p["name"] for p in attn["input_ports"]], ["attn_q", "attn_kv"])
        self.assertEqual([p["name"] for p in attn["output_ports"]], ["attn_out"])

    def test_evidence_order_preserved(self):
        payload = payload_of(full_json())
        node = [n for n in payload["nodes"] if n["stable_id"] == "call:proj@layer0"][0]
        self.assertEqual(node["evidence_refs"], ["fixture:node/proj"])
        group = payload["fold_groups"][0]
        state_mapping = [m for m in group["boundary_mappings"]
                         if m["edge_kind"] == "state"][0]
        self.assertEqual(state_mapping["evidence_refs"],
                         ["fixture:fold/state_a", "fixture:fold/state_b"])

    def test_fold_boundary_mapping_order_preserved(self):
        payload = payload_of(full_json())
        group = payload["fold_groups"][0]
        self.assertEqual([m["edge_kind"] for m in group["boundary_mappings"]],
                         ["data", "state", "select"])
        self.assertEqual([m["external_port"] for m in group["boundary_mappings"]],
                         ["hidden_states", "kv_state", "route_hint"])

    def test_member_node_id_order_preserved(self):
        payload = payload_of(full_json())
        group = payload["fold_groups"][0]
        self.assertEqual(group["member_node_ids"],
                         ["call:proj@layer0", "call:attn@layer0"])


# ---------------------------------------------------------------------------
# 4. 往返：字节级不动点
# ---------------------------------------------------------------------------
class RoundTripFixedPointTest(GraphDocumentTestCase):
    def test_roundtrip_then_reserialize_is_byte_identical(self):
        text = full_json()
        graph = gdoc.graph_from_json(text)
        self.assertEqual(gdoc.graph_to_json(graph), text)

    def test_roundtrip_repeated_is_stable(self):
        text = full_json()
        first = gdoc.graph_to_json(gdoc.graph_from_json(text))
        second = gdoc.graph_to_json(gdoc.graph_from_json(first))
        self.assertEqual(first, second)
        self.assertEqual(first, text)

    def test_roundtrip_of_shuffled_json_normalizes_to_same_bytes(self):
        baseline = full_json()
        payload = payload_of(baseline)
        for collection in COLLECTIONS:
            payload[collection] = list(reversed(payload[collection]))
        shuffled = json.dumps(payload, ensure_ascii=False)
        self.assertNotEqual(shuffled, baseline)
        graph = gdoc.graph_from_json(shuffled)
        self.assertEqual(gdoc.graph_to_json(graph), baseline)

    def test_roundtrip_preserves_entity_contents(self):
        text = full_json()
        graph = gdoc.graph_from_json(text)
        self.assertEqual({n.stable_id for n in graph.nodes},
                         {"call:proj@layer0", "call:attn@layer0"})
        self.assertEqual([p.name for p in
                          [n for n in graph.nodes
                           if n.stable_id == "call:attn@layer0"][0].input_ports],
                         ["attn_q", "attn_kv"])
        self.assertEqual(graph.fold_groups[0].member_node_ids,
                         ("call:proj@layer0", "call:attn@layer0"))

    def test_roundtrip_result_collections_are_canonically_sorted(self):
        payload = payload_of(full_json())
        for collection in COLLECTIONS:
            payload[collection] = list(reversed(payload[collection]))
        graph = gdoc.graph_from_json(json.dumps(payload, ensure_ascii=False))
        node_ids = [n.stable_id for n in graph.nodes]
        self.assertEqual(node_ids, sorted(node_ids))
        guard_pairs = [(a.guard_id, a.scenario_id) for a in graph.guard_activations]
        self.assertEqual(guard_pairs, sorted(guard_pairs))

    def test_roundtrip_across_processes_byte_identical(self):
        text = full_json()
        literal = repr(text)
        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.graph_document import graph_from_json, graph_to_json\n"
            f"text = {literal}\n"
            "sys.stdout.write(graph_to_json(graph_from_json(text)))\n"
        )
        for seed in ("0", "1", "12345"):
            with self.subTest(pythonhashseed=seed):
                proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                      text=True, env=dict(os.environ, PYTHONHASHSEED=seed),
                                      timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout, text)

    def test_importing_graph_document_pulls_no_model_runtimes(self):
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "import structure_reader.graph_document\n"
            "print(json.dumps([m for m in ('torch','transformers','jsonschema','safetensors')"
            " if m in sys.modules]))\n"
        )
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), [])


# ---------------------------------------------------------------------------
# 5. 输入边界：非字符串 / 非法文本 / 顶层 / schema / 缺字段
# ---------------------------------------------------------------------------
class DeserializationBoundaryTest(GraphDocumentTestCase):
    def test_non_string_input_is_controlled_value_error(self):
        for bad in (None, 5, 1.5, True, b"{}", ["{}"], {"schema": SCHEMA}, object()):
            with self.subTest(bad=type(bad).__name__):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: gdoc.graph_from_json(b))
                self.assertIn("字符串", str(exc))

    def test_invalid_json_text_is_controlled_value_error(self):
        for bad in ("", "{", "not json", "{}extra", "{'a':1}"):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: gdoc.graph_from_json(b), "JSON")

    def test_top_level_must_be_json_object(self):
        for bad in ("[]", "5", '"text"', "null", "true"):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: gdoc.graph_from_json(b), "顶层", "对象")

    def test_schema_mismatch_named(self):
        payload = payload_of(full_json())
        payload["schema"] = "semantic-graph/v2"
        self.assertValueErrorLocated(
            lambda: gdoc.graph_from_json(json.dumps(payload)), "schema")
        del payload["schema"]
        self.assertValueErrorLocated(
            lambda: gdoc.graph_from_json(json.dumps(payload)), "schema")

    def test_each_missing_collection_is_named(self):
        for collection in COLLECTIONS:
            payload = payload_of(full_json())
            del payload[collection]
            with self.subTest(collection=collection):
                exc = self.assertValueErrorLocated(
                    lambda p=payload: gdoc.graph_from_json(json.dumps(p)),
                    collection, "缺少必需集合")
                self.assertIn(collection, str(exc))

    def test_all_missing_collections_reported_together(self):
        payload = {"schema": SCHEMA, "template_id": TEMPLATE_ID, "world_size": 1}
        exc = self.assertValueErrorLocated(
            lambda: gdoc.graph_from_json(json.dumps(payload)), "缺少必需集合")
        for collection in COLLECTIONS:
            with self.subTest(collection=collection):
                self.assertIn(collection, str(exc))

    def test_missing_world_size_is_named_not_defaulted(self):
        payload = payload_of(full_json())
        del payload["world_size"]
        exc = self.assertValueErrorLocated(
            lambda: gdoc.graph_from_json(json.dumps(payload)), "world_size")
        self.assertIn("显式", str(exc))

    def test_missing_template_id_rejected(self):
        payload = payload_of(full_json())
        del payload["template_id"]
        self.assertValueErrorLocated(
            lambda: gdoc.graph_from_json(json.dumps(payload)), "template_id")

    def test_collections_must_be_json_arrays(self):
        for collection in COLLECTIONS:
            for bad in (None, "x", 5, {"a": 1}, True):
                payload = payload_of(full_json())
                payload[collection] = bad
                with self.subTest(collection=collection, bad=bad):
                    exc = self.assertValueErrorLocated(
                        lambda p=payload: gdoc.graph_from_json(json.dumps(p)),
                        collection, "JSON 数组")
                    self.assertIn(collection, str(exc))

    def test_entity_item_errors_are_located_with_collection_and_index(self):
        payload = payload_of(full_json())
        payload["nodes"][1] = {"stable_id": "call:broken@layer0"}
        exc = self.assertValueErrorLocated(
            lambda: gdoc.graph_from_json(json.dumps(payload)), "nodes[1]")
        self.assertIn("nodes[1]", str(exc))

    def test_entity_items_of_wrong_json_type_are_controlled(self):
        for collection in COLLECTIONS:
            payload = payload_of(full_json())
            payload[collection] = [5, "x"]
            with self.subTest(collection=collection):
                self.assert_no_bare_exception(
                    lambda p=payload: gdoc.graph_from_json(json.dumps(p)))

    def test_malformed_documents_never_raise_bare_exceptions(self):
        cases = [
            lambda: gdoc.graph_from_json(None),
            lambda: gdoc.graph_from_json(5),
            lambda: gdoc.graph_from_json(b"{}"),
            lambda: gdoc.graph_from_json("{"),
            lambda: gdoc.graph_from_json("[]"),
            lambda: gdoc.graph_from_json("{}"),
        ]
        payload = payload_of(full_json())
        for collection in COLLECTIONS:
            broken = dict(payload)
            broken[collection] = None
            cases.append(lambda b=broken: gdoc.graph_from_json(json.dumps(b)))
            broken2 = dict(payload)
            broken2[collection] = [None]
            cases.append(lambda b=broken2: gdoc.graph_from_json(json.dumps(b)))
        for index, case in enumerate(cases):
            with self.subTest(case=index):
                self.assert_no_bare_exception(case)


# ---------------------------------------------------------------------------
# 6. 容器入口实体类型闸门（构造器）
# ---------------------------------------------------------------------------
class ConstructorEntityTypeTest(GraphDocumentTestCase):
    def test_wrong_entity_type_named_with_collection_and_index(self):
        expected = {"named_axes": "NamedAxis", "tensors": "Tensor", "nodes": "Node",
                    "edges": "Edge", "state_versions": "StateVersion",
                    "state_transitions": "StateTransition", "guards": "Guard",
                    "guard_activations": "GuardActivation",
                    "alias_relations": "AliasRelation", "fold_groups": "FoldGroup"}
        for collection, type_name in expected.items():
            with self.subTest(collection=collection):
                exc = self.assertValueErrorLocated(
                    lambda c=collection: gdoc.SemanticGraph(
                        **graph_kwargs(**{c: [object()]})),
                    f"{collection}[0]", type_name)
                self.assertIn(type_name, str(exc))

    def test_wrong_type_detected_at_second_index(self):
        exc = self.assertValueErrorLocated(
            lambda: gdoc.SemanticGraph(**graph_kwargs(nodes=[node_proj(), object()])),
            "nodes[1]")
        self.assertIn("nodes[1]", str(exc))

    def test_collections_must_be_list_or_tuple(self):
        for collection in COLLECTIONS:
            for bad in (None, "x", 5, {"a": 1}, {1, 2}):
                kwargs = graph_kwargs()
                kwargs[collection] = bad
                with self.subTest(collection=collection, bad=type(bad).__name__):
                    self.assertValueErrorLocated(
                        lambda k=kwargs: gdoc.SemanticGraph(**k), collection)

    def test_valid_entities_accepted_for_every_collection(self):
        graph = full_graph()
        for collection in COLLECTIONS:
            with self.subTest(collection=collection):
                self.assertGreaterEqual(len(getattr(graph, collection)), 1)


# ---------------------------------------------------------------------------
# 7. 重复 stable_id / 重复组合键：两条路径都受控拒绝
# ---------------------------------------------------------------------------
class DuplicateKeyTest(GraphDocumentTestCase):
    def _duplicate_pairs(self):
        return {
            "named_axes": (axis_B(), axis_B()),
            "tensors": (tensor_hidden(), tensor_hidden()),
            "nodes": (node_proj(), node_proj()),
            "edges": (edge_data(), edge_data()),
            "state_versions": (state_versions()[0], state_versions()[0]),
            "state_transitions": (state_transition(), state_transition()),
            "guards": (guard_causal(), guard_causal()),
            "alias_relations": (alias_key(), alias_key()),
            "fold_groups": (fold_group(), fold_group()),
        }

    def test_duplicate_stable_id_rejected_by_constructor_path_serialization(self):
        for collection, pair in self._duplicate_pairs().items():
            with self.subTest(collection=collection):
                kwargs = graph_kwargs(**{collection: list(pair)})
                graph = gdoc.SemanticGraph(**kwargs)
                exc = self.assertValueErrorLocated(
                    lambda g=graph: gdoc.graph_to_json(g), f"{collection}[1]", "重复")
                self.assertIn(collection, str(exc))

    def test_duplicate_stable_id_rejected_directly_by_from_json(self):
        baseline = payload_of(full_json())
        for collection in COLLECTIONS:
            if collection == "guard_activations":
                continue
            payload = payload_of(full_json())
            payload[collection] = [baseline[collection][0], baseline[collection][0]]
            with self.subTest(collection=collection):
                exc = self.assertValueErrorLocated(
                    lambda p=payload: gdoc.graph_from_json(json.dumps(p)),
                    f"{collection}[1]", "重复")
                self.assertIn(collection, str(exc))

    def test_duplicate_guard_activation_pair_rejected_on_both_paths(self):
        first = activation(guard_id="guard:causal", scenario_id="scenario:prefill")
        second = activation(guard_id="guard:causal", scenario_id="scenario:prefill",
                            active_for_scenario="false")
        graph = gdoc.SemanticGraph(**graph_kwargs(guard_activations=[first, second]))
        self.assertValueErrorLocated(
            lambda: gdoc.graph_to_json(graph), "guard_activations[1]", "组合键")
        payload = payload_of(full_json())
        payload["guard_activations"] = [first.to_dict(), second.to_dict()]
        self.assertValueErrorLocated(
            lambda: gdoc.graph_from_json(json.dumps(payload)),
            "guard_activations[1]", "组合键")

    def test_duplicate_detection_reports_actual_duplicate_id(self):
        graph = gdoc.SemanticGraph(**graph_kwargs(
            guards=[guard_causal(), guard_causal(stable_id="guard:causal")]))
        exc = self.assertValueErrorLocated(lambda: gdoc.graph_to_json(graph),
                                           "guard:causal")
        self.assertIn("guard:causal", str(exc))

    def test_distinct_stable_ids_never_rejected(self):
        graph = full_graph(guards=[guard_causal(),
                                   guard_causal(stable_id="guard:window")])
        payload = payload_of(gdoc.graph_to_json(graph))
        self.assertEqual([g["stable_id"] for g in payload["guards"]],
                         ["guard:causal", "guard:window"])

    def test_composite_key_is_two_tuple_not_concatenation(self):
        """返修点 4：含 \\u0000 的字符串不得发生拼接碰撞。"""
        left = activation(guard_id="guard:a\x00b", scenario_id="c")
        right = activation(guard_id="guard:a", scenario_id="b\x00c")
        # 旧实现用 guard_id + NUL + scenario_id 拼串，两者会碰撞
        self.assertNotEqual((left.guard_id, left.scenario_id),
                            (right.guard_id, right.scenario_id))
        self.assertEqual(left.guard_id + "\x00" + left.scenario_id,
                         right.guard_id + "\x00" + right.scenario_id)
        graph = gdoc.SemanticGraph(**graph_kwargs(guard_activations=[left, right]))
        text = gdoc.graph_to_json(graph)
        payload = payload_of(text)
        self.assertEqual(len(payload["guard_activations"]), 2)
        self.assertEqual(sorted((a["guard_id"], a["scenario_id"])
                                for a in payload["guard_activations"]),
                         sorted([(left.guard_id, left.scenario_id),
                                 (right.guard_id, right.scenario_id)]))
        self.assertEqual(gdoc.graph_to_json(gdoc.graph_from_json(text)), text)

    def test_two_tuple_key_still_rejects_true_duplicate_with_nul(self):
        shared = {"guard_id": "guard:a\x00b", "scenario_id": "c"}
        graph = gdoc.SemanticGraph(**graph_kwargs(
            guard_activations=[activation(**shared), activation(**shared)]))
        self.assertValueErrorLocated(lambda: gdoc.graph_to_json(graph),
                                     "guard_activations[1]", "组合键")

    def test_composite_key_sorting_is_tuple_order(self):
        activations = [
            activation(guard_id="guard:b", scenario_id="scenario:1"),
            activation(guard_id="guard:a\x00b", scenario_id="c"),
            activation(guard_id="guard:a", scenario_id="b\x00c"),
            activation(guard_id="guard:a", scenario_id="scenario:1"),
        ]
        graph = gdoc.SemanticGraph(**graph_kwargs(guard_activations=activations))
        payload = payload_of(gdoc.graph_to_json(graph))
        pairs = [(a["guard_id"], a["scenario_id"]) for a in payload["guard_activations"]]
        self.assertEqual(pairs, sorted(
            [(a.guard_id, a.scenario_id) for a in activations]))


# ---------------------------------------------------------------------------
# 8. 公共 API 类型闸门
# ---------------------------------------------------------------------------
class PublicApiGateTest(GraphDocumentTestCase):
    def test_graph_to_json_requires_semantic_graph(self):
        for bad in (None, 5, "graph", {}, [], object()):
            with self.subTest(bad=type(bad).__name__):
                self.assertValueErrorLocated(
                    lambda b=bad: gdoc.graph_to_json(b), "SemanticGraph")

    def test_public_api_surface_is_named_as_documented(self):
        for name in ("SemanticGraph", "graph_to_json", "graph_from_json"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(gdoc, name))


# ---------------------------------------------------------------------------
# 9. 图级语义缺口（本单不通过、不声称）
# ---------------------------------------------------------------------------
class GraphLevelSemanticsGapsTest(GraphDocumentTestCase):
    def test_dangling_node_endpoints_not_checked(self):
        broken = edge_data(src_node_id="call:missing@layer0",
                           dst_node_id="call:also_missing@layer0")
        text = full_json(edges=[broken])
        self.assertEqual(gdoc.graph_to_json(gdoc.graph_from_json(text)), text)

    def test_port_existence_not_checked(self):
        broken = node_proj(input_ports=[gir.Port(name="ghost_in", direction="input",
                                                 tensor_id="tensor:ghost")],
                           output_ports=[])
        text = full_json(nodes=[broken])
        self.assertIn("ghost_in", text)
        self.assertEqual(gdoc.graph_to_json(gdoc.graph_from_json(text)), text)

    def test_axis_and_version_references_not_checked(self):
        transition = state_transition(before_version_id="statever:absent",
                                      after_version_id="statever:absent_too")
        text = full_json(state_transitions=[transition])
        self.assertIn("statever:absent", text)
        self.assertEqual(gdoc.graph_to_json(gdoc.graph_from_json(text)), text)

    def test_alias_tensor_and_guard_ref_existence_not_checked(self):
        alias = alias_key(base_tensor_id="tensor:ghost", alias_tensor_id="tensor:ghost")
        payload = payload_of(full_json(alias_relations=[alias]))
        self.assertEqual(payload["alias_relations"][0]["base_tensor_id"], "tensor:ghost")
        guardless = activation(guard_id="guard:not_defined")
        payload2 = payload_of(full_json(guard_activations=[guardless]))
        self.assertEqual(payload2["guard_activations"][0]["guard_id"], "guard:not_defined")

    def test_no_connectivity_or_semantic_verification_fields_in_document(self):
        payload = payload_of(full_json())
        for forbidden in ("verified", "connectivity_equivalent", "is_valid",
                          "semantic_checked", "errors", "diagnostics"):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, payload)

    def test_duplicate_boundary_key_across_groups_not_checked_at_document_level(self):
        g1 = fold_group(stable_id="fold:g1")
        g2 = fold_group(stable_id="fold:g2",
                        member_node_ids=["call:proj@layer0", "call:attn@layer0"])
        text = full_json(fold_groups=[g1, g2])
        payload = payload_of(text)
        self.assertEqual(len(payload["fold_groups"]), 2)
        self.assertEqual(gdoc.graph_to_json(gdoc.graph_from_json(text)), text)

    def test_serializable_document_is_not_claimed_correct(self):
        """文档可序列化 ≠ 图正确：单元素集合空图也能序列化并往返。"""
        graph = gdoc.SemanticGraph(template_id="probe")
        text = gdoc.graph_to_json(graph)
        self.assertEqual(gdoc.graph_to_json(gdoc.graph_from_json(text)), text)
        self.assertEqual(len(payload_of(text)["nodes"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
