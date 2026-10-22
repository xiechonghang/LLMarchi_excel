"""S1 01a 折叠边界独立验证：`FoldBoundaryMapping` 与 `FoldGroup` 的局部数据契约。

范围严格限定于 `results/development_01a_fold_boundary_probe.md` 声明的增量：
- `FoldBoundaryMapping`（6 字段）：`direction`（限 `input|output`，外部视角）、`external_port`、
  `edge_kind`（限 `data|state|select|control`，**四类边界一律显式映射，不得静默省略**）、
  `internal_node_id`、`internal_port`（五必填字符串）+ 有序 `evidence_refs`（缺省空）；
  `boundary_key()` = `(direction, external_port, edge_kind)`；`to_dict` 恰 6 键；
  `from_dict` 拒绝非对象/缺字段（点名）/`evidence_refs` 非 JSON 数组。
- `FoldGroup`（3 字段）：显式 `stable_id`、有序 `member_node_ids`、有序 `boundary_mappings`；
  局部不变量——组内节点 ID 不得重复；每条映射的 `internal_node_id` 必须属于本组；
  同一外部边界键 **至多一条映射，出现第二次即拒**（不区分是否完全相同）；`to_dict` 恰 3 键。

**明确属图级检查、本单不通过也不声称**（见 `FoldGroupGraphLevelGapsTest`）：
外部 Edge 存在性、旁路边存在性与保留正确性、实际端口引用存在性、折叠前后连通性等价、
跨组边界键冲突、同一节点归属多个组。

不在本范围：Guard/GuardActivation、AliasRelation、状态身份、`SemanticGraph`、
`graph_to_json`/`graph_from_json`、图级 canonical 序列化、01b 检查器、01c schema、
求值器（开发单 02）、源码核验（开发单 03）、真实 DeepSeek 图。

纪律：数据为**合成契约样例（synthetic contract fixture）**，非 DeepSeek 结构；期望值来自任务单与
开发报告声明的接口，不取自产品输出（不复制产品输出为 golden）；只用标准库；不安装/导入
jsonschema、torch、transformers、safetensors。
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, MISSING, fields
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import structure_reader.graph_ir as gir  # noqa: E402

MAPPING_FIELDS = ("direction", "external_port", "edge_kind", "internal_node_id",
                  "internal_port", "evidence_refs")
MAPPING_REQUIRED = ("direction", "external_port", "edge_kind", "internal_node_id",
                    "internal_port")
MAPPING_TO_DICT_KEYS = set(MAPPING_FIELDS)
GROUP_FIELDS = ("stable_id", "member_node_ids", "boundary_mappings")
GROUP_TO_DICT_KEYS = set(GROUP_FIELDS)
DIRECTIONS = ("input", "output")
EDGE_KINDS = ("data", "state", "select", "control")
CANONICAL_KWARGS = {"ensure_ascii": False, "sort_keys": True, "separators": (",", ":")}

NODE_CSA = "call:csa@layer0"
NODE_NORM = "call:norm@layer0"
NODE_ROUTER = "call:router@layer0"
MEMBERS = (NODE_CSA, NODE_NORM, NODE_ROUTER)


def mapping(**overrides):
    """合成契约样例：data 类输入边界。"""
    spec = {
        "direction": "input",
        "external_port": "hidden_states",
        "edge_kind": "data",
        "internal_node_id": NODE_NORM,
        "internal_port": "norm_in",
        "evidence_refs": ["fixture:src:in"],
    }
    spec.update(overrides)
    return gir.FoldBoundaryMapping(**spec)


def scenario_mappings():
    """任务单要求的三类边界：data 输入、state 输出、select 旁路。"""
    return [
        gir.FoldBoundaryMapping(direction="input", external_port="hidden_states",
                                edge_kind="data", internal_node_id=NODE_NORM,
                                internal_port="norm_in",
                                evidence_refs=["fixture:src:in"]),
        gir.FoldBoundaryMapping(direction="output", external_port="kv_state",
                                edge_kind="state", internal_node_id=NODE_CSA,
                                internal_port="kv_out",
                                evidence_refs=["fixture:src:state_a",
                                               "fixture:src:state_b"]),
        gir.FoldBoundaryMapping(direction="input", external_port="route_hint",
                                edge_kind="select", internal_node_id=NODE_ROUTER,
                                internal_port="sel_in",
                                evidence_refs=["fixture:src:sel"]),
    ]


def group(**overrides):
    """合成契约样例：折叠组 + 三类边界映射。"""
    spec = {
        "stable_id": "fold:csa_block@layer0",
        "member_node_ids": list(MEMBERS),
        "boundary_mappings": scenario_mappings(),
    }
    spec.update(overrides)
    return gir.FoldGroup(**spec)


def canonical(item):
    """仅用于稳定性比较：对 to_dict() 做确定性 JSON 编码（非产品 API）。"""
    return json.dumps(item.to_dict(), **CANONICAL_KWARGS)


class FoldTestCase(unittest.TestCase):
    def assert_located(self, exc, *needles):
        message = str(exc)
        self.assertTrue(
            any(needle in message for needle in needles),
            f"错误信息未定位到 {needles!r}；实际信息：{message!r}",
        )

    def assertValueErrorLocated(self, callable_, *needles):
        with self.assertRaises(ValueError) as ctx:
            callable_()
        self.assert_located(ctx.exception, *needles)
        return ctx.exception


# ---------------------------------------------------------------------------
# 1. FoldBoundaryMapping 接口快照
# ---------------------------------------------------------------------------
class FoldMappingInterfaceTest(FoldTestCase):
    def test_mapping_is_frozen_dataclass_with_declared_field_order(self):
        self.assertTrue(gir.FoldBoundaryMapping.__dataclass_params__.frozen)
        self.assertEqual(tuple(f.name for f in fields(gir.FoldBoundaryMapping)),
                         MAPPING_FIELDS)

    def test_required_fields_and_evidence_default(self):
        by_name = {f.name: f for f in fields(gir.FoldBoundaryMapping)}
        for required in MAPPING_REQUIRED:
            with self.subTest(required=required):
                self.assertIs(by_name[required].default, MISSING)
        self.assertEqual(by_name["evidence_refs"].default, ())
        self.assertEqual(mapping().evidence_refs, ("fixture:src:in",))
        self.assertEqual(mapping(evidence_refs=[]).evidence_refs, ())

    def test_to_dict_has_exactly_six_keys(self):
        payload = mapping().to_dict()
        self.assertEqual(set(payload), MAPPING_TO_DICT_KEYS)
        self.assertEqual(len(payload), 6)
        self.assertEqual(payload["edge_kind"], "data")
        self.assertEqual(payload["direction"], "input")

    def test_boundary_key_is_derived_tuple_of_three(self):
        item = mapping(direction="output", external_port="kv_state", edge_kind="state")
        key = item.boundary_key()
        self.assertIsInstance(key, tuple)
        self.assertEqual(key, ("output", "kv_state", "state"))
        self.assertEqual(len(key), 3)

    def test_boundary_key_is_deterministic_and_not_a_field(self):
        item = mapping()
        self.assertEqual(item.boundary_key(), item.boundary_key())
        field_names = {f.name for f in fields(gir.FoldBoundaryMapping)}
        self.assertNotIn("boundary_key", field_names)
        self.assertNotIn("boundary_key", item.to_dict())

    def test_frozen_and_hashable(self):
        item = mapping()
        for attribute, value in (("direction", "output"), ("external_port", "other"),
                                 ("edge_kind", "state"), ("internal_port", "p"),
                                 ("evidence_refs", ())):
            with self.subTest(attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(item, attribute, value)
        self.assertEqual(hash(item), hash(mapping()))
        self.assertEqual(len({item, mapping()}), 1)

    def test_unexpected_constructor_keyword_rejected(self):
        with self.assertRaises(TypeError):
            gir.FoldBoundaryMapping(direction="input", external_port="p", edge_kind="data",
                                    internal_node_id=NODE_NORM, internal_port="in",
                                    internal_node="n")

    def test_mapping_has_no_graph_level_resolution_fields(self):
        """局部类不携带图级核验结果字段。"""
        field_names = {f.name for f in fields(gir.FoldBoundaryMapping)}
        for forbidden in ("resolved", "exists", "external_edge_id", "target_node_id",
                          "bypass_preserved", "connected"):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, field_names)
        payload = mapping().to_dict()
        for forbidden in ("resolved", "exists", "bypass_preserved"):
            with self.subTest(payload_field=forbidden):
                self.assertNotIn(forbidden, payload)

    def test_non_mapping_input_rejected_by_group(self):
        """组内元素类型闸门（映射侧接口）。"""
        with self.assertRaises(ValueError):
            group(boundary_mappings=[mapping().to_dict()])


# ---------------------------------------------------------------------------
# 2. direction：input / output
# ---------------------------------------------------------------------------
class FoldMappingDirectionTest(FoldTestCase):
    def test_input_and_output_both_accepted_and_preserved(self):
        for direction in DIRECTIONS:
            with self.subTest(direction=direction):
                item = mapping(direction=direction)
                self.assertEqual(item.direction, direction)
                self.assertEqual(item.to_dict()["direction"], direction)
                self.assertEqual(item.boundary_key()[0], direction)
                self.assertEqual(gir.FoldBoundaryMapping.from_dict(item.to_dict()), item)

    def test_direction_is_case_sensitive(self):
        for bad in ("Input", "INPUT", "Output", "OUTPUT", "In"):
            with self.subTest(direction=bad):
                exc = self.assertValueErrorLocated(lambda b=bad: mapping(direction=b),
                                                   "direction")
                self.assertIn(repr(bad), str(exc))

    def test_invalid_direction_values_rejected_with_field_name(self):
        for bad in (None, 5, True, ["input"], {"direction": "input"}):
            with self.subTest(direction=bad):
                self.assertValueErrorLocated(lambda b=bad: mapping(direction=b), "direction")

    def test_direction_error_lists_both_legal_values(self):
        exc = self.assertValueErrorLocated(lambda: mapping(direction="inout"), "direction")
        for value in DIRECTIONS:
            self.assertIn(value, str(exc))
        self.assertIn("inout", str(exc))

    def test_direction_empty_and_whitespace_use_enum_error_not_text_error(self):
        """边界观察：direction 不走通用文本规则，空/空白由枚举闸门具名拒绝。"""
        for bad in ("", "   ", " input", "input "):
            with self.subTest(direction=bad):
                exc = self.assertValueErrorLocated(lambda b=bad: mapping(direction=b),
                                                   "direction")
                self.assertIn("必须是", str(exc))
                self.assertIn(repr(bad), str(exc))

    def test_direction_whitelist_precedes_evidence_gate(self):
        exc = self.assertValueErrorLocated(
            lambda: mapping(direction="bogus", evidence_refs="nope"), "direction")
        self.assertIn("bogus", str(exc))


# ---------------------------------------------------------------------------
# 3. edge_kind：data / state / select / control
# ---------------------------------------------------------------------------
class FoldMappingEdgeKindTest(FoldTestCase):
    def test_all_four_edge_kinds_accepted_and_not_silently_omitted(self):
        for kind in EDGE_KINDS:
            with self.subTest(edge_kind=kind):
                item = mapping(edge_kind=kind)
                self.assertEqual(item.edge_kind, kind)
                self.assertEqual(item.to_dict()["edge_kind"], kind)
                self.assertEqual(item.boundary_key()[2], kind)
                self.assertEqual(gir.FoldBoundaryMapping.from_dict(item.to_dict()), item)

    def test_state_select_control_are_distinct_kept_facts(self):
        items = [mapping(edge_kind=kind, external_port=f"port_{kind}")
                 for kind in EDGE_KINDS]
        self.assertEqual(len({canonical(item) for item in items}), 4)
        self.assertEqual({item.to_dict()["edge_kind"] for item in items}, set(EDGE_KINDS))

    def test_edge_kind_is_case_sensitive(self):
        for bad in ("Data", "DATA", "State", "Select", "Control", "Control "):
            with self.subTest(edge_kind=bad):
                exc = self.assertValueErrorLocated(lambda b=bad: mapping(edge_kind=b),
                                                   "edge_kind")
                self.assertIn(repr(bad), str(exc))

    def test_invalid_edge_kind_values_rejected_with_field_name(self):
        for bad in (None, 5, True, ["data"], {"edge_kind": "data"}):
            with self.subTest(edge_kind=bad):
                self.assertValueErrorLocated(lambda b=bad: mapping(edge_kind=b), "edge_kind")

    def test_edge_kind_error_lists_all_four_legal_values(self):
        exc = self.assertValueErrorLocated(lambda: mapping(edge_kind="bypass"), "edge_kind")
        for kind in EDGE_KINDS:
            self.assertIn(kind, str(exc))
        self.assertIn("bypass", str(exc))

    def test_edge_kind_empty_and_whitespace_use_enum_error(self):
        for bad in ("", "   ", " data"):
            with self.subTest(edge_kind=bad):
                exc = self.assertValueErrorLocated(lambda b=bad: mapping(edge_kind=b),
                                                   "edge_kind")
                self.assertIn(repr(bad), str(exc))

    def test_edge_kind_whitelist_precedes_evidence_gate(self):
        exc = self.assertValueErrorLocated(
            lambda: mapping(edge_kind="bogus", evidence_refs="nope"), "edge_kind")
        self.assertIn("bogus", str(exc))

    def test_control_mapping_is_constructible_and_roundtrips(self):
        item = mapping(direction="output", external_port="ctrl_ready",
                       edge_kind="control", internal_node_id=NODE_ROUTER,
                       internal_port="ctrl_out", evidence_refs=["fixture:src:ctrl"])
        self.assertEqual(item.edge_kind, "control")
        self.assertEqual(gir.FoldBoundaryMapping.from_dict(item.to_dict()), item)


# ---------------------------------------------------------------------------
# 4. 必填文本与有序 evidence_refs
# ---------------------------------------------------------------------------
class FoldMappingTextAndEvidenceTest(FoldTestCase):
    def test_text_fields_require_nonempty_string(self):
        for field_name in ("external_port", "internal_node_id", "internal_port"):
            for bad in ("", "   ", "\t", None, 5, True, ["p"]):
                with self.subTest(field=field_name, bad=bad):
                    self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: mapping(**{f: b}), field_name
                    )

    def test_text_fields_reject_surrounding_whitespace(self):
        for field_name in ("external_port", "internal_node_id", "internal_port"):
            for bad in (f" {field_name}", f"{field_name} ", f"\t{field_name}"):
                with self.subTest(field=field_name, bad=bad):
                    exc = self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: mapping(**{f: b}), field_name
                    )
                    self.assertIn("空白", str(exc))

    def test_evidence_default_empty_and_roundtrips(self):
        item = mapping(evidence_refs=[])
        self.assertEqual(item.evidence_refs, ())
        self.assertEqual(item.to_dict()["evidence_refs"], [])
        self.assertEqual(gir.FoldBoundaryMapping.from_dict(item.to_dict()).evidence_refs, ())

    def test_evidence_order_preserved(self):
        refs = ["fixture:src:a", "fixture:src:b", "fixture:src:c"]
        item = mapping(evidence_refs=refs)
        self.assertEqual(list(item.evidence_refs), refs)
        self.assertEqual(item.to_dict()["evidence_refs"], refs)
        back = gir.FoldBoundaryMapping.from_dict(item.to_dict())
        self.assertEqual(list(back.evidence_refs), refs)
        reversed_item = mapping(evidence_refs=list(reversed(refs)))
        self.assertNotEqual(canonical(item), canonical(reversed_item))

    def test_evidence_container_must_be_list_or_tuple(self):
        for bad in ({"fixture:a"}, {"fixture:a": 1}, (x for x in ["fixture:a"])):
            with self.subTest(bad=type(bad).__name__):
                exc = self.assertValueErrorLocated(lambda b=bad: mapping(evidence_refs=b),
                                                   "evidence_refs")
                self.assertIn("list/tuple", str(exc))
                self.assertIn(type(bad).__name__, str(exc))

    def test_evidence_string_not_split_and_dict_keys_not_taken(self):
        exc = self.assertValueErrorLocated(lambda: mapping(evidence_refs="fixture:a"),
                                           "evidence_refs")
        self.assertIn("str", str(exc))
        exc_dict = self.assertValueErrorLocated(
            lambda: mapping(evidence_refs={"fixture:a": 1}), "evidence_refs")
        self.assertIn("dict", str(exc_dict))

    def test_evidence_items_must_be_nonempty_strings_with_index(self):
        for bad, needle in (([None], "evidence_refs[0]"), ([""], "evidence_refs[0]"),
                            (["   "], "evidence_refs[0]"), ([5], "evidence_refs[0]"),
                            ([[]], "evidence_refs[0]"),
                            (["ok", None], "evidence_refs[1]"),
                            (["ok", "", "x"], "evidence_refs[1]")):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(lambda b=bad: mapping(evidence_refs=b), needle)

    def test_evidence_frozen_to_tuple_with_defensive_copy(self):
        refs = ["fixture:src:a"]
        item = mapping(evidence_refs=refs)
        self.assertIsInstance(item.evidence_refs, tuple)
        refs.append("fixture:src:b")
        self.assertEqual(item.evidence_refs, ("fixture:src:a",))

    def test_evidence_duplicates_allowed(self):
        item = mapping(evidence_refs=["fixture:same", "fixture:same"])
        self.assertEqual(item.evidence_refs, ("fixture:same", "fixture:same"))

    def test_evidence_not_part_of_boundary_key(self):
        first = mapping(evidence_refs=["fixture:a"])
        second = mapping(evidence_refs=["fixture:b"])
        self.assertEqual(first.boundary_key(), second.boundary_key())
        self.assertNotEqual(first, second)


# ---------------------------------------------------------------------------
# 5. FoldBoundaryMapping 往返、JSON 数组闸门、字节稳定
# ---------------------------------------------------------------------------
class FoldMappingRoundTripTest(FoldTestCase):
    def test_roundtrip_equality_and_dict_equality(self):
        for item in scenario_mappings():
            with self.subTest(edge_kind=item.edge_kind):
                back = gir.FoldBoundaryMapping.from_dict(item.to_dict())
                self.assertEqual(back, item)
                self.assertEqual(back.to_dict(), item.to_dict())

    def test_repeated_canonical_serialization_byte_identical(self):
        for item in scenario_mappings():
            with self.subTest(edge_kind=item.edge_kind):
                first = canonical(item)
                self.assertEqual(first.encode("utf-8"), canonical(item).encode("utf-8"))
                self.assertEqual(
                    first,
                    canonical(gir.FoldBoundaryMapping.from_dict(item.to_dict())),
                )

    def test_canonical_bytes_deterministic_across_processes(self):
        item = mapping(evidence_refs=["fixture:中文", "fixture:b"])
        expected = canonical(item)
        literal = repr(json.dumps(item.to_dict(), ensure_ascii=False))
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.graph_ir import FoldBoundaryMapping\n"
            f"data = json.loads({literal})\n"
            "m = FoldBoundaryMapping.from_dict(data)\n"
            "sys.stdout.write(json.dumps(m.to_dict(), ensure_ascii=False, sort_keys=True,"
            " separators=(',', ':')))\n"
        )
        for seed in ("0", "1", "12345"):
            with self.subTest(pythonhashseed=seed):
                proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                      text=True, env=dict(os.environ, PYTHONHASHSEED=seed),
                                      timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout, expected)

    def test_from_dict_requires_json_object(self):
        for bad in ([], "mapping", 3, None, True):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: gir.FoldBoundaryMapping.from_dict(b), "折叠边界映射", "对象"
                )

    def test_from_dict_missing_required_fields_named(self):
        for missing in MAPPING_REQUIRED:
            payload = mapping().to_dict()
            del payload[missing]
            with self.subTest(missing=missing):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.FoldBoundaryMapping.from_dict(p), missing
                )

    def test_from_dict_evidence_refs_must_be_json_array(self):
        for bad in (None, 5, True, "ev", "", {"fixture:a": 1}):
            payload = mapping().to_dict()
            payload["evidence_refs"] = bad
            with self.subTest(bad=bad):
                exc = self.assertValueErrorLocated(
                    lambda p=payload: gir.FoldBoundaryMapping.from_dict(p), "evidence_refs"
                )
                self.assertIn("JSON 数组", str(exc))

    def test_from_dict_enforces_enum_invariants(self):
        payload = mapping().to_dict()
        payload["direction"] = "sideways"
        self.assertValueErrorLocated(
            lambda: gir.FoldBoundaryMapping.from_dict(payload), "direction")
        payload2 = mapping().to_dict()
        payload2["edge_kind"] = "bypass"
        self.assertValueErrorLocated(
            lambda: gir.FoldBoundaryMapping.from_dict(payload2), "edge_kind")

    def test_from_dict_optional_evidence_absent_defaults_empty(self):
        payload = {"direction": "input", "external_port": "p", "edge_kind": "data",
                   "internal_node_id": NODE_NORM, "internal_port": "in"}
        item = gir.FoldBoundaryMapping.from_dict(payload)
        self.assertEqual(item.evidence_refs, ())

    def test_from_dict_extra_keys_ignored(self):
        payload = mapping().to_dict()
        payload["resolved"] = True
        payload["external_edge_id"] = "edge:1"
        self.assertEqual(gir.FoldBoundaryMapping.from_dict(payload), mapping())
        self.assertEqual(gir.FoldBoundaryMapping.from_dict(payload).to_dict(),
                         mapping().to_dict())

    def test_distinct_mappings_do_not_collapse(self):
        a = mapping(external_port="p1")
        b = mapping(external_port="p2")
        c = mapping(external_port="p1", direction="output")
        d = mapping(external_port="p1", edge_kind="state")
        self.assertEqual(len({a, b, c, d}), 4)


# ---------------------------------------------------------------------------
# 6. FoldGroup 接口快照
# ---------------------------------------------------------------------------
class FoldGroupInterfaceTest(FoldTestCase):
    def test_group_is_frozen_dataclass_with_declared_field_order(self):
        self.assertTrue(gir.FoldGroup.__dataclass_params__.frozen)
        self.assertEqual(tuple(f.name for f in fields(gir.FoldGroup)), GROUP_FIELDS)

    def test_all_three_group_fields_are_required(self):
        by_name = {f.name: f for f in fields(gir.FoldGroup)}
        for required in GROUP_FIELDS:
            with self.subTest(required=required):
                self.assertIs(by_name[required].default, MISSING)

    def test_stable_id_is_explicit_field_not_derived_property(self):
        self.assertNotIsInstance(inspect.getattr_static(gir.FoldGroup, "stable_id", None),
                                 property)
        self.assertEqual(group(stable_id="g0").stable_id, "g0")

    def test_to_dict_has_exactly_three_keys_and_six_key_mappings(self):
        payload = group().to_dict()
        self.assertEqual(set(payload), GROUP_TO_DICT_KEYS)
        self.assertEqual(len(payload), 3)
        self.assertEqual(len(payload["boundary_mappings"]), 3)
        for nested in payload["boundary_mappings"]:
            with self.subTest(external_port=nested["external_port"]):
                self.assertEqual(set(nested), MAPPING_TO_DICT_KEYS)
                self.assertEqual(len(nested), 6)

    def test_members_and_mappings_frozen_to_tuples_with_defensive_copy(self):
        members = list(MEMBERS)
        mappings = scenario_mappings()
        item = gir.FoldGroup(stable_id="g", member_node_ids=members,
                             boundary_mappings=mappings)
        self.assertIsInstance(item.member_node_ids, tuple)
        self.assertIsInstance(item.boundary_mappings, tuple)
        members.append("call:extra@layer0")
        mappings.append(mapping(external_port="extra"))
        self.assertEqual(item.member_node_ids, MEMBERS)
        self.assertEqual(len(item.boundary_mappings), 3)

    def test_frozen_and_hashable(self):
        item = group()
        for attribute, value in (("stable_id", "other"), ("member_node_ids", ()),
                                 ("boundary_mappings", ())):
            with self.subTest(attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(item, attribute, value)
        self.assertEqual(hash(item), hash(group()))
        self.assertEqual(len({item, group()}), 1)

    def test_unexpected_constructor_keyword_rejected(self):
        with self.assertRaises(TypeError):
            gir.FoldGroup(stable_id="g", member_node_ids=[], boundary_mappings=[],
                          members=[])

    def test_empty_members_and_mappings_allowed(self):
        item = gir.FoldGroup(stable_id="fold:empty", member_node_ids=[],
                             boundary_mappings=[])
        self.assertEqual(item.member_node_ids, ())
        self.assertEqual(item.boundary_mappings, ())
        self.assertEqual(gir.FoldGroup.from_dict(item.to_dict()), item)


# ---------------------------------------------------------------------------
# 7. 节点成员关系与容器闸门
# ---------------------------------------------------------------------------
class FoldGroupMembershipTest(FoldTestCase):
    def test_stable_id_rules(self):
        for bad in ("", "   ", "\tg"):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(lambda b=bad: group(stable_id=b), "stable_id")
        for bad in (None, 5, True, ["g"]):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(lambda b=bad: group(stable_id=b), "stable_id")

    def test_member_node_ids_container_gate(self):
        for bad in ("node", {"node"}, {"node": 1}, 5, None):
            with self.subTest(bad=type(bad).__name__):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: group(member_node_ids=b), "member_node_ids")
                self.assertIn("list/tuple", str(exc))

    def test_member_node_items_must_be_nonempty_strings_with_index(self):
        for bad, needle in (([None], "member_node_ids[0]"), ([""], "member_node_ids[0]"),
                            (["  "], "member_node_ids[0]"), ([5], "member_node_ids[0]"),
                            ([NODE_CSA, ""], "member_node_ids[1]")):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(lambda b=bad: group(member_node_ids=b), needle)

    def test_member_node_items_reject_surrounding_whitespace(self):
        for bad in (f" {NODE_CSA}", f"{NODE_CSA} "):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(lambda b=bad: group(member_node_ids=[bad]),
                                             "member_node_ids[0]")

    def test_duplicate_member_node_ids_rejected(self):
        exc = self.assertValueErrorLocated(
            lambda: group(member_node_ids=[NODE_CSA, NODE_NORM, NODE_CSA]),
            "重复", "member_node_ids")
        self.assertIn(NODE_CSA, str(exc))

    def test_member_order_preserved(self):
        item = group(member_node_ids=[NODE_ROUTER, NODE_CSA, NODE_NORM])
        self.assertEqual(item.member_node_ids, (NODE_ROUTER, NODE_CSA, NODE_NORM))
        self.assertEqual(item.to_dict()["member_node_ids"],
                         [NODE_ROUTER, NODE_CSA, NODE_NORM])
        self.assertEqual(
            list(gir.FoldGroup.from_dict(item.to_dict()).member_node_ids),
            [NODE_ROUTER, NODE_CSA, NODE_NORM],
        )

    def test_boundary_mappings_container_gate(self):
        for bad in ("m", {"m"}, 5, None, {"external_port": "p"}):
            with self.subTest(bad=type(bad).__name__):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: group(boundary_mappings=b), "boundary_mappings")
                self.assertIn("list/tuple", str(exc))

    def test_boundary_mapping_items_must_be_mapping_instances_with_index(self):
        for bad, needle in (([None], "boundary_mappings[0]"),
                            ([{"direction": "input"}], "boundary_mappings[0]"),
                            ([mapping(), "x"], "boundary_mappings[1]")):
            with self.subTest(bad=bad):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: group(boundary_mappings=b), needle)
                self.assertIn("FoldBoundaryMapping", str(exc))

    def test_mapping_internal_node_must_belong_to_group(self):
        outsider = mapping(internal_node_id="call:not_a_member@layer0")
        exc = self.assertValueErrorLocated(
            lambda: group(boundary_mappings=[outsider]), "不属于本折叠组",
            "not_a_member")
        self.assertIn("boundary_mappings[0]", str(exc))

    def test_mapping_membership_checked_per_index(self):
        good = mapping(external_port="p1")
        bad = mapping(external_port="p2", internal_node_id="call:elsewhere@layer0")
        exc = self.assertValueErrorLocated(
            lambda: group(boundary_mappings=[good, bad]), "boundary_mappings[1]")
        self.assertIn("不属于本折叠组", str(exc))

    def test_membership_uses_member_set_not_group_stable_id(self):
        """映射归属看成员集合；换成另一组稳定 ID 不影响判定。"""
        item = group(stable_id="fold:renamed", boundary_mappings=[mapping()])
        self.assertEqual(len(item.boundary_mappings), 1)
        exc = self.assertValueErrorLocated(
            lambda: group(stable_id="fold:renamed",
                          member_node_ids=[NODE_CSA],
                          boundary_mappings=[mapping()]),
            "不属于本折叠组")
        self.assertIn(NODE_NORM, str(exc))


# ---------------------------------------------------------------------------
# 8. 重复外部边界键的受控拒绝（局部唯一性）
# ---------------------------------------------------------------------------
class FoldGroupBoundaryKeyUniquenessTest(FoldTestCase):
    def test_exact_duplicate_mapping_rejected(self):
        first = mapping(external_port="p", internal_port="a")
        second = mapping(external_port="p", internal_port="a")
        self.assertEqual(first, second)
        exc = self.assertValueErrorLocated(
            lambda: group(boundary_mappings=[first, second]), "重复映射", "p")
        self.assertIn("input", str(exc))
        self.assertIn("data", str(exc))

    def test_same_key_different_target_rejected(self):
        first = mapping(external_port="p", internal_port="a", internal_node_id=NODE_NORM)
        second = mapping(external_port="p", internal_port="b", internal_node_id=NODE_CSA)
        self.assertNotEqual(first, second)
        exc = self.assertValueErrorLocated(
            lambda: group(boundary_mappings=[first, second]), "重复映射")
        self.assertIn("p", str(exc))

    def test_duplicate_key_reported_at_second_occurrence(self):
        a = mapping(external_port="p1")
        b = mapping(external_port="p2")
        dup = mapping(external_port="p1", internal_port="other")
        self.assertValueErrorLocated(
            lambda: group(boundary_mappings=[a, b, dup]), "重复映射")

    def test_same_external_port_with_different_direction_allowed(self):
        inbound = mapping(direction="input", external_port="p", internal_port="a")
        outbound = mapping(direction="output", external_port="p", internal_port="b")
        item = group(boundary_mappings=[inbound, outbound])
        self.assertEqual(len(item.boundary_mappings), 2)
        self.assertNotEqual(inbound.boundary_key(), outbound.boundary_key())

    def test_same_external_port_with_different_edge_kind_allowed(self):
        data_rel = mapping(external_port="p", edge_kind="data", internal_port="a")
        state_rel = mapping(external_port="p", edge_kind="state", internal_port="b")
        select_rel = mapping(external_port="p", edge_kind="select", internal_port="c")
        control_rel = mapping(external_port="p", edge_kind="control", internal_port="d")
        item = group(boundary_mappings=[data_rel, state_rel, select_rel, control_rel])
        self.assertEqual(len(item.boundary_mappings), 4)
        self.assertEqual(len({m.boundary_key() for m in item.boundary_mappings}), 4)

    def test_each_key_component_alone_distinguishes(self):
        base = mapping(direction="input", external_port="p", edge_kind="data")
        self.assertNotEqual(base.boundary_key(),
                            mapping(direction="output", external_port="p",
                                    edge_kind="data").boundary_key())
        self.assertNotEqual(base.boundary_key(),
                            mapping(direction="input", external_port="q",
                                    edge_kind="data").boundary_key())
        self.assertNotEqual(base.boundary_key(),
                            mapping(direction="input", external_port="p",
                                    edge_kind="state").boundary_key())

    def test_four_distinct_data_boundaries_allowed(self):
        items = [mapping(direction=("input" if i % 2 == 0 else "output"),
                         external_port=f"p{i}", edge_kind="data") for i in range(4)]
        item = group(boundary_mappings=items)
        self.assertEqual(len(item.boundary_mappings), 4)
        self.assertEqual(
            [m.external_port for m in item.boundary_mappings],
            ["p0", "p1", "p2", "p3"],
        )

    def test_uniqueness_is_per_key_not_per_mapping_identity(self):
        """不同证据的两条映射若键相同仍拒。"""
        first = mapping(external_port="p", evidence_refs=["fixture:a"])
        second = mapping(external_port="p", evidence_refs=["fixture:b"])
        self.assertNotEqual(first, second)
        self.assertEqual(first.boundary_key(), second.boundary_key())
        self.assertValueErrorLocated(
            lambda: group(boundary_mappings=[first, second]), "重复映射")

    def test_duplicate_detection_independent_of_mapping_order(self):
        first = mapping(external_port="p", internal_port="a")
        second = mapping(external_port="p", internal_port="b")
        self.assertValueErrorLocated(
            lambda: group(boundary_mappings=[first, second]), "重复映射")
        self.assertValueErrorLocated(
            lambda: group(boundary_mappings=[second, first]), "重复映射")

    def test_order_of_valid_mappings_preserved_not_reordered(self):
        items = scenario_mappings() + [mapping(direction="output",
                                               external_port="ctrl_ready",
                                               edge_kind="control",
                                               internal_node_id=NODE_ROUTER,
                                               internal_port="ctrl_out")]
        item = group(boundary_mappings=items)
        self.assertEqual([(m.direction, m.external_port, m.edge_kind)
                          for m in item.boundary_mappings],
                         [("input", "hidden_states", "data"),
                          ("output", "kv_state", "state"),
                          ("input", "route_hint", "select"),
                          ("output", "ctrl_ready", "control")])


# ---------------------------------------------------------------------------
# 9. 三类边界（data 输入 / state 输出 / select 旁路）序列化后均保留
# ---------------------------------------------------------------------------
class FoldGroupScenarioTest(FoldTestCase):
    def test_three_required_boundary_categories_survive_serialization(self):
        item = group()
        payload = item.to_dict()
        kinds = {(m["direction"], m["edge_kind"]) for m in payload["boundary_mappings"]}
        self.assertEqual(kinds, {("input", "data"), ("output", "state"),
                                 ("input", "select")})
        back = gir.FoldGroup.from_dict(payload)
        self.assertEqual(back, item)
        self.assertEqual({(m.direction, m.edge_kind) for m in back.boundary_mappings},
                         {("input", "data"), ("output", "state"), ("input", "select")})

    def test_select_bypass_is_mapped_explicitly_not_dropped(self):
        item = group()
        select_mappings = [m for m in item.boundary_mappings if m.edge_kind == "select"]
        self.assertEqual(len(select_mappings), 1)
        self.assertEqual(select_mappings[0].external_port, "route_hint")
        self.assertEqual(select_mappings[0].internal_node_id, NODE_ROUTER)
        payload = item.to_dict()
        self.assertTrue(any(m["edge_kind"] == "select" for m in payload["boundary_mappings"]))

    def test_state_output_boundary_keeps_evidence_order(self):
        item = group()
        state_mappings = [m for m in item.boundary_mappings if m.edge_kind == "state"]
        self.assertEqual(len(state_mappings), 1)
        self.assertEqual(list(state_mappings[0].evidence_refs),
                         ["fixture:src:state_a", "fixture:src:state_b"])
        payload = item.to_dict()
        state_payload = [m for m in payload["boundary_mappings"]
                         if m["edge_kind"] == "state"][0]
        self.assertEqual(state_payload["evidence_refs"],
                         ["fixture:src:state_a", "fixture:src:state_b"])

    def test_data_input_boundary_points_at_member_node_port(self):
        item = group()
        data_mappings = [m for m in item.boundary_mappings if m.edge_kind == "data"]
        self.assertEqual(len(data_mappings), 1)
        self.assertEqual(data_mappings[0].direction, "input")
        self.assertIn(data_mappings[0].internal_node_id, item.member_node_ids)
        self.assertEqual(data_mappings[0].internal_port, "norm_in")

    def test_all_four_edge_kinds_in_one_group_roundtrip(self):
        items = [mapping(direction="input", external_port="p_data", edge_kind="data"),
                 mapping(direction="output", external_port="p_state", edge_kind="state",
                         internal_node_id=NODE_CSA),
                 mapping(direction="input", external_port="p_select", edge_kind="select",
                         internal_node_id=NODE_ROUTER),
                 mapping(direction="output", external_port="p_control",
                         edge_kind="control", internal_node_id=NODE_ROUTER)]
        item = group(boundary_mappings=items)
        back = gir.FoldGroup.from_dict(item.to_dict())
        self.assertEqual(back, item)
        self.assertEqual([m.edge_kind for m in back.boundary_mappings],
                         ["data", "state", "select", "control"])

    def test_scenario_canonical_bytes_stable(self):
        item = group()
        first = canonical(item)
        self.assertEqual(first, canonical(group()))
        self.assertEqual(first, canonical(gir.FoldGroup.from_dict(item.to_dict())))

    def test_scenario_canonical_bytes_deterministic_across_processes(self):
        item = group()
        expected = canonical(item)
        literal = repr(json.dumps(item.to_dict(), ensure_ascii=False))
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.graph_ir import FoldGroup\n"
            f"data = json.loads({literal})\n"
            "g = FoldGroup.from_dict(data)\n"
            "sys.stdout.write(json.dumps(g.to_dict(), ensure_ascii=False, sort_keys=True,"
            " separators=(',', ':')))\n"
        )
        for seed in ("0", "1", "12345"):
            with self.subTest(pythonhashseed=seed):
                proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                      text=True, env=dict(os.environ, PYTHONHASHSEED=seed),
                                      timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout, expected)

    def test_nested_payload_shape_is_plain_json(self):
        payload = group().to_dict()
        text = json.dumps(payload, ensure_ascii=False)
        reloaded = json.loads(text)
        self.assertEqual(reloaded, payload)
        self.assertEqual(gir.FoldGroup.from_dict(reloaded), group())

    def test_group_does_not_import_model_runtimes(self):
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "import structure_reader.graph_ir\n"
            "print(json.dumps([m for m in ('torch','transformers','jsonschema','safetensors')"
            " if m in sys.modules]))\n"
        )
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), [])


# ---------------------------------------------------------------------------
# 10. FoldGroup 往返与 JSON 数组边界
# ---------------------------------------------------------------------------
class FoldGroupRoundTripTest(FoldTestCase):
    def test_roundtrip_equality_and_dict_equality(self):
        item = group()
        back = gir.FoldGroup.from_dict(item.to_dict())
        self.assertEqual(back, item)
        self.assertEqual(back.to_dict(), item.to_dict())

    def test_repeated_canonical_serialization_byte_identical(self):
        item = group()
        self.assertEqual(canonical(item).encode("utf-8"), canonical(item).encode("utf-8"))

    def test_from_dict_requires_json_object(self):
        for bad in ([], "group", 3, None, True):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: gir.FoldGroup.from_dict(b), "折叠组", "对象"
                )

    def test_from_dict_missing_required_fields_named(self):
        for missing in GROUP_FIELDS:
            payload = group().to_dict()
            del payload[missing]
            with self.subTest(missing=missing):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.FoldGroup.from_dict(p), missing
                )

    def test_from_dict_member_and_mapping_must_be_json_arrays(self):
        for key in ("member_node_ids", "boundary_mappings"):
            for bad in (None, "x", 5, {"a": 1}):
                payload = group().to_dict()
                payload[key] = bad
                with self.subTest(key=key, bad=bad):
                    exc = self.assertValueErrorLocated(
                        lambda p=payload: gir.FoldGroup.from_dict(p), key)
                    self.assertIn("JSON 数组", str(exc))

    def test_from_dict_nested_mapping_errors_named(self):
        payload = group().to_dict()
        del payload["boundary_mappings"][0]["internal_port"]
        self.assertValueErrorLocated(
            lambda: gir.FoldGroup.from_dict(payload), "折叠边界映射记录缺少字段",
            "internal_port")
        payload2 = group().to_dict()
        payload2["boundary_mappings"][1] = "not-a-mapping"
        self.assertValueErrorLocated(
            lambda: gir.FoldGroup.from_dict(payload2), "折叠边界映射记录必须是 JSON 对象")

    def test_from_dict_enforces_group_invariants(self):
        payload = group().to_dict()
        payload["member_node_ids"] = [NODE_CSA, NODE_CSA]
        self.assertValueErrorLocated(
            lambda: gir.FoldGroup.from_dict(payload), "重复")
        payload2 = group().to_dict()
        payload2["boundary_mappings"][0]["internal_node_id"] = "call:outsider@layer0"
        self.assertValueErrorLocated(
            lambda: gir.FoldGroup.from_dict(payload2), "不属于本折叠组")
        payload3 = group().to_dict()
        payload3["boundary_mappings"].append(
            dict(payload3["boundary_mappings"][0]))
        self.assertValueErrorLocated(
            lambda: gir.FoldGroup.from_dict(payload3), "重复映射")

    def test_from_dict_extra_keys_ignored(self):
        payload = group().to_dict()
        payload["collapsed_ok"] = True
        payload["boundary_mappings"][0]["resolved"] = True
        self.assertEqual(gir.FoldGroup.from_dict(payload), group())

    def test_distinct_groups_do_not_collapse(self):
        a = group(stable_id="g1")
        b = group(stable_id="g2")
        c = group(stable_id="g1", member_node_ids=[NODE_CSA],
                  boundary_mappings=[mapping(internal_node_id=NODE_CSA)])
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len({a, b, c}), 3)


# ---------------------------------------------------------------------------
# 11. 图级缺口（本单不通过、不声称）
# ---------------------------------------------------------------------------
class FoldGroupGraphLevelGapsTest(FoldTestCase):
    def test_external_edge_existence_not_checked(self):
        item = mapping(external_port="port_that_may_not_exist")
        self.assertEqual(item.external_port, "port_that_may_not_exist")
        self.assertEqual(gir.FoldBoundaryMapping.from_dict(item.to_dict()), item)

    def test_internal_port_existence_not_checked(self):
        item = mapping(internal_port="no_such_port")
        self.assertEqual(item.internal_port, "no_such_port")
        self.assertEqual(gir.FoldBoundaryMapping.from_dict(item.to_dict()), item)

    def test_connectivity_equivalence_not_verified(self):
        """图级缺口：折叠前后连通性等价、旁路是否被正确保留，本地不判断。"""
        item = group()
        payload = item.to_dict()
        for forbidden in ("connectivity_equivalent", "bypass_preserved",
                          "collapsed_edge_count", "verified"):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, payload)
        self.assertEqual(gir.FoldGroup.from_dict(payload), item)

    def test_same_node_may_belong_to_multiple_groups_locally(self):
        first = group(stable_id="g1", boundary_mappings=[mapping()])
        second = group(stable_id="g2", boundary_mappings=[mapping()])
        self.assertIn(NODE_NORM, first.member_node_ids)
        self.assertIn(NODE_NORM, second.member_node_ids)
        self.assertNotEqual(first, second)

    def test_same_boundary_key_allowed_across_different_groups(self):
        first = group(stable_id="g1", boundary_mappings=[mapping(external_port="shared")])
        second = group(stable_id="g2", boundary_mappings=[mapping(external_port="shared")])
        self.assertEqual(first.boundary_mappings[0].boundary_key(),
                         second.boundary_mappings[0].boundary_key())
        self.assertNotEqual(first, second)

    def test_group_stable_id_uniqueness_not_enforced(self):
        a = group(stable_id="dup")
        b = group(stable_id="dup")
        self.assertEqual(a, b)
        self.assertEqual(len([a, b]), 2)

    def test_mapping_target_node_need_not_be_a_known_graph_node(self):
        """仅当节点属于本组成员集合即可；是否为真实图节点本地不查。"""
        item = group(member_node_ids=["call:invented@layer0"],
                     boundary_mappings=[mapping(internal_node_id="call:invented@layer0")])
        self.assertEqual(len(item.boundary_mappings), 1)
        self.assertEqual(gir.FoldGroup.from_dict(item.to_dict()), item)

    def test_duplicate_pair_ordering_is_deterministic_report_only(self):
        """重复键的错误信息给出先出现者与后出现者，供定位；不代表图级冲突解决。"""
        first = mapping(external_port="p", internal_port="a")
        second = mapping(external_port="p", internal_port="b")
        exc = self.assertValueErrorLocated(
            lambda: group(boundary_mappings=[first, second]), "重复映射")
        self.assertIn("a", str(exc))
        self.assertIn("b", str(exc))


if __name__ == "__main__":
    unittest.main(verbosity=2)
