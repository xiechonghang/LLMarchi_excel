"""S1 01a Edge 探针独立验证：稳定 ID、四种 kind、端点与 tensor_id、state 双向规则、往返与 frozen。

范围严格限定于 `results/development_01a_edge_probe.md` 声明的 Edge 增量：
- 显式 `stable_id`（必填字段，非派生）；端点 `(src_node_id, src_port) → (dst_node_id, dst_port)`、
  `tensor_id`（张量 stable_id，不是显示名）均为必填非空字符串、拒绝首尾空白；
- `kind` 限 `data|state|select|control`，大小写敏感，非法值回显收到的值；
- **state 双向规则**：`state` 边必须携带 `state_transition_ref`；`data`/`select`/`control`
  边禁止携带；`guard_ref` 可选（无 guard 对象，不校验指向）；
- `evidence_refs` 有序、仅接受 list/tuple、各项非空字符串（字符串不拆解、dict 不取键）；
- `to_dict` 恰 10 键、`from_dict` 稳定往返（dataclass 相等、重复与跨进程字节一致）；frozen。

**明确属图级检查、本单不通过也不声称**（见 `EdgeGraphLevelGapsTest`）：
端点存在性、端口方向一致性、`tensor_id`/`guard_ref`/`state_transition_ref` 指向的实体存在性、
跨调用版本顺序与无版本循环、图级 `stable_id` 唯一性、按语义轴/布局判连接相容。

不在本范围：SemanticGraph、`graph_to_json`/`graph_from_json`、图级 canonical 序列化、
guard 三态对象、state before/after 版本对象、alias/fold/template-instance、01b 检查器、
01c schema、求值器（开发单 02）、源码核验（开发单 03）、真实 DeepSeek 结构提取。

纪律：数据为**合成契约样例（synthetic contract fixture）**，非 DeepSeek 结构；
期望值来自任务单与开发报告声明的接口，不取自产品输出（不复制产品输出为 golden）；
只用标准库；不安装/导入 jsonschema、torch、transformers、safetensors。
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

KINDS = ("data", "state", "select", "control")
REQUIRED_FIELDS = ("stable_id", "src_node_id", "src_port", "dst_node_id", "dst_port",
                   "tensor_id", "kind")
EDGE_TO_DICT_KEYS = {
    "stable_id", "src_node_id", "src_port", "dst_node_id", "dst_port", "tensor_id",
    "kind", "evidence_refs", "guard_ref", "state_transition_ref",
}
CANONICAL_KWARGS = {"ensure_ascii": False, "sort_keys": True, "separators": (",", ":")}


def edge(**overrides):
    """合成契约样例：data 边（q_proj → attn，张量 tensor:q）。"""
    spec = {
        "stable_id": "edge:q_out->attn_in@layer0",
        "src_node_id": "call:q_proj@layer0",
        "src_port": "q_out",
        "dst_node_id": "call:attn@layer0",
        "dst_port": "q_in",
        "tensor_id": "tensor:q",
        "kind": "data",
        "evidence_refs": ["fixture:edge/ev1", "fixture:edge/ev2"],
    }
    spec.update(overrides)
    return gir.Edge(**spec)


def state_edge(**overrides):
    """合成契约样例：state 边（跨调用 KV 缓存 before→after）。"""
    spec = {
        "stable_id": "edge:kv_state@layer1",
        "src_node_id": "call:attn@layer0",
        "src_port": "kv_out",
        "dst_node_id": "call:attn@layer1",
        "dst_port": "kv_in",
        "tensor_id": "tensor:kv",
        "kind": "state",
        "state_transition_ref": "state:kv_cache/before->after",
    }
    spec.update(overrides)
    return gir.Edge(**spec)


def canonical(item):
    """仅用于稳定性比较：对 to_dict() 做确定性 JSON 编码（非产品 API）。"""
    return json.dumps(item.to_dict(), **CANONICAL_KWARGS)


class EdgeTestCase(unittest.TestCase):
    """公共断言辅助。"""

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
# 1. 接口快照：字段、必填/可选、显式 stable_id、frozen、序列化键集
# ---------------------------------------------------------------------------
class EdgeInterfaceSnapshotTest(EdgeTestCase):
    def test_edge_is_frozen_dataclass_with_declared_field_order(self):
        self.assertTrue(gir.Edge.__dataclass_params__.frozen)
        self.assertEqual(
            tuple(f.name for f in fields(gir.Edge)),
            ("stable_id", "src_node_id", "src_port", "dst_node_id", "dst_port",
             "tensor_id", "kind", "evidence_refs", "guard_ref", "state_transition_ref"),
        )

    def test_required_fields_and_optional_defaults(self):
        by_name = {f.name: f for f in fields(gir.Edge)}
        for required in REQUIRED_FIELDS:
            with self.subTest(required=required):
                self.assertIs(by_name[required].default, MISSING)
        self.assertEqual(by_name["evidence_refs"].default, ())
        self.assertIsNone(by_name["guard_ref"].default)
        self.assertIsNone(by_name["state_transition_ref"].default)
        minimal = gir.Edge(stable_id="e0", src_node_id="n0", src_port="out",
                           dst_node_id="n1", dst_port="in", tensor_id="tensor:t",
                           kind="data")
        self.assertEqual(minimal.evidence_refs, ())
        self.assertIsNone(minimal.guard_ref)
        self.assertIsNone(minimal.state_transition_ref)

    def test_stable_id_is_explicit_field_not_derived_property(self):
        static = inspect.getattr_static(gir.Edge, "stable_id", None)
        self.assertNotIsInstance(static, property)
        self.assertEqual(edge(stable_id="plain_edge_id").stable_id, "plain_edge_id")

    def test_frozen_prevents_mutation_and_is_hashable(self):
        item = edge()
        for attribute, new_value in (("stable_id", "other"), ("kind", "state"),
                                     ("tensor_id", "tensor:other"), ("evidence_refs", ())):
            with self.subTest(attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(item, attribute, new_value)
        self.assertEqual(hash(item), hash(edge()))
        self.assertEqual(len({item, edge()}), 1)

    def test_evidence_refs_frozen_to_tuple_with_defensive_copy(self):
        refs = ["fixture:edge/ev1"]
        item = edge(evidence_refs=refs)
        self.assertIsInstance(item.evidence_refs, tuple)
        refs.append("fixture:edge/ev3")
        self.assertEqual(item.evidence_refs, ("fixture:edge/ev1",))

    def test_to_dict_has_exactly_ten_keys(self):
        payload = edge().to_dict()
        self.assertEqual(set(payload), EDGE_TO_DICT_KEYS)
        self.assertEqual(len(payload), 10)

    def test_unexpected_constructor_keyword_rejected(self):
        with self.assertRaises(TypeError):
            gir.Edge(stable_id="e", src_node_id="a", src_port="o", dst_node_id="b",
                     dst_port="i", tensor_id="tensor:t", kind="data",
                     edge_id="e")


# ---------------------------------------------------------------------------
# 2. kind：四种取值、大小写敏感、非法值
# ---------------------------------------------------------------------------
class EdgeKindTest(EdgeTestCase):
    def test_four_kinds_accepted_and_roundtrip(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                item = state_edge(kind=kind) if kind == "state" else edge(kind=kind)
                self.assertEqual(item.kind, kind)
                self.assertEqual(item.to_dict()["kind"], kind)
                self.assertEqual(gir.Edge.from_dict(item.to_dict()), item)

    def test_kind_is_case_sensitive(self):
        """大小写敏感：非小写合法值走白名单拒绝并回显收到的值。"""
        for bad in ("Data", "DATA", "State", "STATE", "Select", "Control"):
            with self.subTest(kind=bad):
                exc = self.assertValueErrorLocated(lambda b=bad: edge(kind=b), "kind")
                self.assertIn(repr(bad), str(exc))  # 回显收到的值

    def test_kind_surrounding_whitespace_rejected_before_whitelist(self):
        """首尾空白先被通用文本规则拒绝（同为字段级 kind 错误，早于白名单分支）。"""
        for bad in ("select ", " control", "data\n", "\tdata"):
            with self.subTest(kind=bad):
                exc = self.assertValueErrorLocated(lambda b=bad: edge(kind=b), "kind")
                self.assertIn("空白", str(exc))

    def test_kind_error_lists_all_four_legal_values(self):
        exc = self.assertValueErrorLocated(lambda: edge(kind="inout"), "kind")
        for kind in KINDS:
            self.assertIn(kind, str(exc))
        self.assertIn("inout", str(exc))

    def test_invalid_kind_values_rejected_with_field_name(self):
        for bad in ("", "   ", None, 5, True, ["data"], {"kind": "data"}):
            with self.subTest(kind=bad):
                self.assertValueErrorLocated(lambda b=bad: edge(kind=b), "kind")

    def test_kind_validation_precedes_state_reference_rule(self):
        """kind 非法 + 携带 state_transition_ref 时，错误应指向 kind（先校验 kind）。"""
        exc = self.assertValueErrorLocated(
            lambda: edge(kind="bogus", state_transition_ref="state:x/before->after"), "kind"
        )
        self.assertIn("bogus", str(exc))

    def test_each_kind_is_a_distinct_fact(self):
        items = []
        for kind in KINDS:
            items.append(state_edge(kind=kind) if kind == "state" else edge(kind=kind))
        self.assertEqual(len(items), 4)
        self.assertEqual({item.kind for item in items}, set(KINDS))
        self.assertEqual(len({canonical(item) for item in items}), 4)


# ---------------------------------------------------------------------------
# 3. 端点与 tensor_id
# ---------------------------------------------------------------------------
class EdgeEndpointAndTensorTest(EdgeTestCase):
    def test_endpoints_and_tensor_id_preserved_verbatim(self):
        item = edge()
        payload = item.to_dict()
        self.assertEqual(payload["src_node_id"], "call:q_proj@layer0")
        self.assertEqual(payload["src_port"], "q_out")
        self.assertEqual(payload["dst_node_id"], "call:attn@layer0")
        self.assertEqual(payload["dst_port"], "q_in")
        self.assertEqual(payload["tensor_id"], "tensor:q")
        back = gir.Edge.from_dict(payload)
        self.assertEqual(back, item)
        self.assertEqual((back.src_node_id, back.src_port, back.dst_node_id, back.dst_port),
                         ("call:q_proj@layer0", "q_out", "call:attn@layer0", "q_in"))

    def test_required_text_fields_must_be_nonempty_strings(self):
        for field_name in REQUIRED_FIELDS:
            for bad in ("", "   ", "\t\n", None, 123, b"x"):
                with self.subTest(field=field_name, bad=bad):
                    self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: edge(**{f: b}), field_name
                    )

    def test_required_text_fields_reject_surrounding_whitespace(self):
        for field_name in REQUIRED_FIELDS:
            for bad in (" value", "value ", "\tvalue", "value\n"):
                with self.subTest(field=field_name, bad=bad):
                    exc = self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: edge(**{f: b}), field_name
                    )
                    self.assertIn("空白", str(exc))

    def test_tensor_id_is_an_id_not_a_display_name(self):
        """边界观察：tensor_id 原样保存，无前缀/格式约束（引用存在性属图级）。"""
        item = edge(tensor_id="just-an-opaque-id")
        self.assertEqual(item.tensor_id, "just-an-opaque-id")
        self.assertEqual(gir.Edge.from_dict(item.to_dict()).tensor_id, "just-an-opaque-id")

    def test_missing_endpoint_or_tensor_references_not_checked_locally(self):
        """图级缺口（不算通过）：端点/张量不存在时本类仍接受。"""
        item = edge(src_node_id="node:does_not_exist", dst_node_id="node:also_missing",
                    tensor_id="tensor:missing")
        self.assertEqual(item.src_node_id, "node:does_not_exist")
        self.assertEqual(gir.Edge.from_dict(item.to_dict()), item)

    def test_self_loop_is_locally_representable(self):
        """图级缺口（不算通过）：自环/同端口连接本类不做方向或拓扑拒绝。"""
        item = edge(src_node_id="call:attn@layer0", dst_node_id="call:attn@layer0",
                    src_port="x", dst_port="x")
        self.assertEqual(item.src_node_id, item.dst_node_id)
        self.assertEqual(gir.Edge.from_dict(item.to_dict()), item)

    def test_endpoint_internal_whitespace_allowed(self):
        """边界观察：仅拒绝首尾空白，内部空格允许（与 symbol/端口命名同类宽松度）。"""
        item = edge(src_node_id="call:q proj@layer0")
        self.assertEqual(item.src_node_id, "call:q proj@layer0")


# ---------------------------------------------------------------------------
# 4. 普通边 / 状态边的 state_transition_ref 双向规则
# ---------------------------------------------------------------------------
class EdgeStateTransitionTest(EdgeTestCase):
    def test_state_edge_requires_state_transition_ref(self):
        exc = self.assertValueErrorLocated(
            lambda: state_edge(state_transition_ref=None), "state_transition_ref"
        )
        self.assertIn("state", str(exc))

    def test_state_edge_missing_ref_also_detected_via_from_dict(self):
        payload = state_edge().to_dict()
        payload["state_transition_ref"] = None
        self.assertValueErrorLocated(
            lambda: gir.Edge.from_dict(payload), "state_transition_ref"
        )

    def test_state_edge_with_ref_accepted_and_roundtrip(self):
        item = state_edge()
        self.assertEqual(item.kind, "state")
        self.assertEqual(item.state_transition_ref, "state:kv_cache/before->after")
        payload = item.to_dict()
        self.assertEqual(payload["state_transition_ref"], "state:kv_cache/before->after")
        back = gir.Edge.from_dict(payload)
        self.assertEqual(back, item)
        self.assertEqual(back.state_transition_ref, item.state_transition_ref)

    def test_non_state_kinds_forbid_state_transition_ref(self):
        for kind in ("data", "select", "control"):
            with self.subTest(kind=kind):
                exc = self.assertValueErrorLocated(
                    lambda k=kind: edge(kind=k, state_transition_ref="state:x/before->after"),
                    "state_transition_ref",
                )
                self.assertIn(kind, str(exc))

    def test_non_state_kinds_forbid_ref_also_via_from_dict(self):
        for kind in ("data", "select", "control"):
            payload = edge(kind=kind).to_dict()
            payload["state_transition_ref"] = "state:x/before->after"
            with self.subTest(kind=kind):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.Edge.from_dict(p), "state_transition_ref"
                )

    def test_state_transition_ref_must_be_nonempty_string_or_none(self):
        for bad in ("", "   ", "\t", 5, True, ["state:x"]):
            with self.subTest(ref=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: state_edge(state_transition_ref=b), "state_transition_ref"
                )

    def test_state_transition_ref_rejects_surrounding_whitespace(self):
        for bad in (" state:x/before->after", "state:x/before->after ", "\tstate:x"):
            with self.subTest(ref=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: state_edge(state_transition_ref=b), "state_transition_ref"
                )

    def test_state_edge_and_data_edge_are_distinct_facts(self):
        data_item = edge()
        state_item = state_edge()
        self.assertNotEqual(data_item, state_item)
        self.assertNotEqual(canonical(data_item), canonical(state_item))
        self.assertEqual(data_item.to_dict()["state_transition_ref"], None)
        self.assertIsNotNone(state_item.to_dict()["state_transition_ref"])


# ---------------------------------------------------------------------------
# 5. 可选 guard_ref
# ---------------------------------------------------------------------------
class EdgeGuardRefTest(EdgeTestCase):
    def test_guard_ref_optional_and_roundtrip(self):
        without = edge()
        with_guard = edge(guard_ref="guard:is_prefill")
        self.assertIsNone(without.guard_ref)
        self.assertEqual(with_guard.guard_ref, "guard:is_prefill")
        self.assertEqual(gir.Edge.from_dict(with_guard.to_dict()), with_guard)
        self.assertEqual(gir.Edge.from_dict(with_guard.to_dict()).guard_ref, "guard:is_prefill")
        self.assertNotEqual(without, with_guard)

    def test_guard_ref_rejects_empty_whitespace_and_non_string(self):
        for bad in ("", "   ", "\t", 5, True, ["guard:x"]):
            with self.subTest(guard=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: edge(guard_ref=b), "guard_ref"
                )

    def test_guard_ref_rejects_surrounding_whitespace(self):
        for bad in (" guard:x", "guard:x ", "\tguard:x"):
            with self.subTest(guard=bad):
                self.assertValueErrorLocated(lambda b=bad: edge(guard_ref=b), "guard_ref")

    def test_guard_ref_absent_key_means_none(self):
        payload = edge().to_dict()
        del payload["guard_ref"]
        self.assertIsNone(gir.Edge.from_dict(payload).guard_ref)

    def test_guard_ref_on_state_edge_is_allowed(self):
        """边界观察：未禁止 state 边同时携带 guard_ref（是否为有意义组合属图级语义）。"""
        item = state_edge(guard_ref="guard:cache_valid")
        self.assertEqual(item.kind, "state")
        self.assertEqual(item.guard_ref, "guard:cache_valid")
        self.assertEqual(gir.Edge.from_dict(item.to_dict()), item)


# ---------------------------------------------------------------------------
# 6. 有序 evidence_refs
# ---------------------------------------------------------------------------
class EdgeEvidenceRefsTest(EdgeTestCase):
    def test_default_empty_and_roundtrip(self):
        item = edge(evidence_refs=[])
        self.assertEqual(item.evidence_refs, ())
        self.assertEqual(item.to_dict()["evidence_refs"], [])
        self.assertEqual(gir.Edge.from_dict(item.to_dict()).evidence_refs, ())

    def test_order_preserved(self):
        refs = ["fixture:edge/a", "fixture:edge/b", "fixture:edge/c"]
        item = edge(evidence_refs=refs)
        self.assertEqual(list(item.evidence_refs), refs)
        self.assertEqual(item.to_dict()["evidence_refs"], refs)
        self.assertEqual(list(gir.Edge.from_dict(item.to_dict()).evidence_refs), refs)
        reversed_item = edge(evidence_refs=list(reversed(refs)))
        self.assertNotEqual(canonical(item), canonical(reversed_item))

    def test_items_must_be_nonempty_strings_with_index(self):
        for bad, needle in (([None], "evidence_refs[0]"), ([""], "evidence_refs[0]"),
                            (["   "], "evidence_refs[0]"), ([5], "evidence_refs[0]"),
                            ([[]], "evidence_refs[0]"),
                            (["ok", None], "evidence_refs[1]")):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: edge(evidence_refs=b), needle
                )

    def test_container_must_be_list_or_tuple(self):
        for bad in ({"fixture:a"}, {"fixture:a": 1}, (x for x in ["fixture:a"])):
            with self.subTest(bad=type(bad).__name__):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: edge(evidence_refs=b), "evidence_refs"
                )
                self.assertIn("list/tuple", str(exc))
                self.assertIn(type(bad).__name__, str(exc))

    def test_list_and_tuple_accepted(self):
        self.assertEqual(edge(evidence_refs=["fixture:a"]).evidence_refs, ("fixture:a",))
        self.assertEqual(edge(evidence_refs=("fixture:a",)).evidence_refs, ("fixture:a",))

    def test_duplicates_allowed(self):
        """边界观察：未规定引用去重，重复项原样保留。"""
        item = edge(evidence_refs=["fixture:same", "fixture:same"])
        self.assertEqual(item.evidence_refs, ("fixture:same", "fixture:same"))
        self.assertEqual(gir.Edge.from_dict(item.to_dict()).evidence_refs,
                         ("fixture:same", "fixture:same"))


# ---------------------------------------------------------------------------
# 7. to_dict / from_dict 往返与受控破坏
# ---------------------------------------------------------------------------
class EdgeRoundTripTest(EdgeTestCase):
    def test_data_edge_roundtrip_equality(self):
        item = edge()
        back = gir.Edge.from_dict(item.to_dict())
        self.assertEqual(back, item)
        self.assertEqual(back.to_dict(), item.to_dict())

    def test_state_edge_roundtrip_equality(self):
        item = state_edge(guard_ref="guard:any")
        back = gir.Edge.from_dict(item.to_dict())
        self.assertEqual(back, item)
        self.assertEqual(canonical(back), canonical(item))

    def test_repeated_canonical_serialization_byte_identical(self):
        item = state_edge()
        first = canonical(item)
        self.assertEqual(first.encode("utf-8"), canonical(item).encode("utf-8"))
        self.assertEqual(first, canonical(gir.Edge.from_dict(item.to_dict())))

    def test_canonical_bytes_deterministic_across_processes(self):
        item = state_edge(guard_ref="guard:cache_valid")
        expected = canonical(item)
        literal = repr(json.dumps(item.to_dict(), ensure_ascii=False))
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.graph_ir import Edge\n"
            f"data = json.loads({literal})\n"
            "e = Edge.from_dict(data)\n"
            "sys.stdout.write(json.dumps(e.to_dict(), ensure_ascii=False, sort_keys=True,"
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
        for bad in ([], "edge", 3, None, True):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(lambda b=bad: gir.Edge.from_dict(b),
                                             "边记录", "对象")

    def test_from_dict_missing_required_fields_named(self):
        for missing in REQUIRED_FIELDS:
            payload = edge().to_dict()
            del payload[missing]
            with self.subTest(missing=missing):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.Edge.from_dict(p), missing
                )

    def test_from_dict_evidence_refs_must_be_json_array(self):
        for bad in (None, 5, True, "ev1", "", {"fixture:a": 1}):
            payload = edge().to_dict()
            payload["evidence_refs"] = bad
            with self.subTest(bad=bad):
                exc = self.assertValueErrorLocated(
                    lambda p=payload: gir.Edge.from_dict(p), "evidence_refs"
                )
                self.assertIn("JSON 数组", str(exc))

    def test_from_dict_optional_keys_absent_default(self):
        payload = {"stable_id": "e0", "src_node_id": "n0", "src_port": "out",
                   "dst_node_id": "n1", "dst_port": "in", "tensor_id": "tensor:t",
                   "kind": "data"}
        item = gir.Edge.from_dict(payload)
        self.assertEqual(item.evidence_refs, ())
        self.assertIsNone(item.guard_ref)
        self.assertIsNone(item.state_transition_ref)

    def test_from_dict_extra_keys_ignored(self):
        """边界观察：未知多余键被忽略（前向兼容），不影响往返相等。"""
        payload = edge().to_dict()
        payload["template_id"] = "tpl"
        payload["shape_matches"] = True
        self.assertEqual(gir.Edge.from_dict(payload), edge())
        self.assertEqual(gir.Edge.from_dict(payload).to_dict(), edge().to_dict())

    def test_distinct_edges_do_not_collapse(self):
        a = edge(stable_id="e_a")
        b = edge(stable_id="e_b")
        c = edge(stable_id="e_a", kind="select", src_port="idx_out")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len({a, b, c}), 3)


# ---------------------------------------------------------------------------
# 8. 图级缺口：本单不做、也不算通过（如实固定当前本地宽松行为）
# ---------------------------------------------------------------------------
class EdgeGraphLevelGapsTest(EdgeTestCase):
    def test_graph_level_stable_id_uniqueness_not_enforced(self):
        """图级缺口：同一 stable_id 的两条边在本地均可构造且相等（无登记表）。"""
        first = edge(stable_id="e_dup")
        second = edge(stable_id="e_dup")
        self.assertEqual(first, second)
        self.assertEqual(len([first, second]), 2)

    def test_port_direction_consistency_not_checked(self):
        """图级缺口：src_port/dst_port 的方向是否与节点端口声明一致，本地无法判断。"""
        item = edge(src_port="an_output_port", dst_port="an_input_port")
        self.assertEqual((item.src_port, item.dst_port), ("an_output_port", "an_input_port"))
        self.assertEqual(gir.Edge.from_dict(item.to_dict()), item)

    def test_tensor_reference_existence_not_checked(self):
        """图级缺口：tensor_id 是否指向真实张量，本地无法判断。"""
        item = edge(tensor_id="tensor:not_in_any_graph")
        self.assertEqual(item.tensor_id, "tensor:not_in_any_graph")

    def test_guard_and_state_references_not_resolved(self):
        """图级缺口：guard_ref/state_transition_ref 仅为字符串，指向对象本文件未建模。"""
        item = state_edge(guard_ref="guard:does_not_exist",
                          state_transition_ref="state:does_not_exist/before->after")
        self.assertEqual(gir.Edge.from_dict(item.to_dict()), item)

    def test_no_version_cycle_or_ordering_check(self):
        """图级缺口：before/after 版本方向与无版本循环检查不在本类。"""
        item = state_edge(state_transition_ref="state:kv/after->before")  # 反向也可本地构造
        self.assertEqual(item.state_transition_ref, "state:kv/after->before")
        self.assertEqual(gir.Edge.from_dict(item.to_dict()), item)

    def test_connection_compatibility_by_shape_not_judged(self):
        """图级缺口：不以数值 shape 相等判连接相容（本类无 shape 概念）。"""
        item = edge(tensor_id="tensor:any")
        payload = item.to_dict()
        self.assertNotIn("shape", payload)
        self.assertNotIn("compatible", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
