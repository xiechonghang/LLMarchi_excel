"""S1 01a 状态身份独立验证：StateVersion 稳定身份、StateTransition 方向与证据、往返与 frozen。

范围严格限定于 `results/development_01a_state_identity_probe.md` 声明的两类增量：
- `StateVersion`：显式 `stable_id`（字段，非派生 property）、`state_id`、`version_label`
  三必填字符串，拒绝非字符串/空值/首尾空白；`to_dict` 恰 3 键；`from_dict` 稳定往返；
  **不携带任何 cache 属性**（capacity/valid_length/remainder/overlap/Indexer 均未实现）。
- `StateTransition`：显式 `stable_id`、`state_id`、`before_version_id`、`after_version_id`
  四必填字符串 + 有序可选 `evidence_refs`；**方向保真**（before 与 after 必须不同，
  交换即不同事实，序列化原样保留不重排）；`evidence_refs` 仅接受 list/tuple、
  各项非空字符串（字符串不拆解、dict 不取键）；`to_dict` 恰 5 键；`from_dict` 拒绝
  非对象/缺字段/`evidence_refs` 非 JSON 数组；frozen、可哈希、重复与跨进程字节稳定。
- 交叉约定：构造 `kind="state"` 的 Edge 引用 `StateTransition.stable_id`，验证
  state 边/普通边既有规则仍成立（Edge 侧字段与校验未改）。

**明确属图级检查、本单不通过也不声称**（见 `StateGraphLevelGapsTest` 与
`UnimplementedCacheAttributesTest`）：
- `state_transition_ref` / `before_version_id` / `after_version_id` 指向的实体存在性；
- `StateTransition.state_id` 与其引用版本所属 `state_id` 的一致性；
- 跨调用版本先后与无版本循环；图级 `stable_id` 唯一性；
- capacity、valid_length、压缩 remainder/overlap、Indexer 状态等 cache 数值属性。

**本文件不是完整 cache 测试**：仅覆盖状态版本标识与有向更新关系。

不在本范围：Guard/GuardActivation、SemanticGraph、`graph_to_json`/`graph_from_json`、
图级 canonical 序列化、alias/fold/template-instance、01b 检查器、01c schema、
求值器（开发单 02）、源码核验（开发单 03）、真实 DeepSeek 结构提取。

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

STATE_VERSION_FIELDS = ("stable_id", "state_id", "version_label")
STATE_TRANSITION_REQUIRED = ("stable_id", "state_id", "before_version_id",
                             "after_version_id")
STATE_VERSION_TO_DICT_KEYS = {"stable_id", "state_id", "version_label"}
STATE_TRANSITION_TO_DICT_KEYS = {"stable_id", "state_id", "before_version_id",
                                 "after_version_id", "evidence_refs"}
CANONICAL_KWARGS = {"ensure_ascii": False, "sort_keys": True, "separators": (",", ":")}
# cache 数值属性：本单声明未实现，必须确认两类上都不存在
UNIMPLEMENTED_CACHE_FIELDS = ("capacity", "valid_length", "remainder", "overlap",
                              "indexer", "indexer_state", "block_table", "slot_mapping")


def state_version(**overrides):
    """合成契约样例：KV 缓存在 t0 的 before 版本。"""
    spec = {
        "stable_id": "ver:kv_cache@t0",
        "state_id": "state:kv_cache",
        "version_label": "before",
    }
    spec.update(overrides)
    return gir.StateVersion(**spec)


def state_transition(**overrides):
    """合成契约样例：KV 缓存 step1 的 before→after 有向更新。"""
    spec = {
        "stable_id": "trans:kv_cache@step1",
        "state_id": "state:kv_cache",
        "before_version_id": "ver:kv_cache@t0",
        "after_version_id": "ver:kv_cache@t1",
        "evidence_refs": ["fixture:state/ev1", "fixture:state/ev2"],
    }
    spec.update(overrides)
    return gir.StateTransition(**spec)


def canonical(item):
    """仅用于稳定性比较：对 to_dict() 做确定性 JSON 编码（非产品 API）。"""
    return json.dumps(item.to_dict(), **CANONICAL_KWARGS)


class StateIdentityTestCase(unittest.TestCase):
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
# 1. StateVersion 接口快照：字段、必填、显式 stable_id、frozen、序列化键集
# ---------------------------------------------------------------------------
class StateVersionInterfaceTest(StateIdentityTestCase):
    def test_state_version_is_frozen_dataclass_with_declared_field_order(self):
        self.assertTrue(gir.StateVersion.__dataclass_params__.frozen)
        self.assertEqual(tuple(f.name for f in fields(gir.StateVersion)),
                         STATE_VERSION_FIELDS)

    def test_all_three_fields_are_required_without_defaults(self):
        by_name = {f.name: f for f in fields(gir.StateVersion)}
        for required in STATE_VERSION_FIELDS:
            with self.subTest(required=required):
                self.assertIs(by_name[required].default, MISSING)

    def test_stable_id_is_explicit_field_not_derived_property(self):
        """稳定身份是显式字段：不派生自 state_id/version_label，原样保存。"""
        static = inspect.getattr_static(gir.StateVersion, "stable_id", None)
        self.assertNotIsInstance(static, property)
        item = state_version(stable_id="plain-version-id")
        self.assertEqual(item.stable_id, "plain-version-id")
        self.assertNotEqual(item.stable_id, item.state_id)
        self.assertNotEqual(item.stable_id, item.version_label)

    def test_to_dict_has_exactly_three_keys(self):
        payload = state_version().to_dict()
        self.assertEqual(set(payload), STATE_VERSION_TO_DICT_KEYS)
        self.assertEqual(len(payload), 3)

    def test_frozen_prevents_mutation_and_is_hashable(self):
        item = state_version()
        for attribute, new_value in (("stable_id", "other"), ("state_id", "state:other"),
                                     ("version_label", "after")):
            with self.subTest(attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(item, attribute, new_value)
        self.assertEqual(hash(item), hash(state_version()))
        self.assertEqual(len({item, state_version()}), 1)

    def test_unexpected_constructor_keyword_rejected(self):
        with self.assertRaises(TypeError):
            gir.StateVersion(stable_id="v", state_id="s", version_label="before",
                             version="before")

    def test_no_cache_attributes_exist_on_state_version(self):
        """任务单要求明确：本类只是版本标识，不含 capacity/valid_length 等 cache 属性。"""
        field_names = {f.name for f in fields(gir.StateVersion)}
        for name in UNIMPLEMENTED_CACHE_FIELDS:
            with self.subTest(attribute=name):
                self.assertNotIn(name, field_names)
                self.assertFalse(hasattr(state_version(), name))
        self.assertEqual(len(field_names), 3)


# ---------------------------------------------------------------------------
# 2. StateVersion 身份语义：state_id/version_label 保真与区分
# ---------------------------------------------------------------------------
class StateVersionIdentityTest(StateIdentityTestCase):
    def test_identity_fields_preserved_verbatim_and_roundtrip(self):
        item = state_version()
        payload = item.to_dict()
        self.assertEqual(payload["stable_id"], "ver:kv_cache@t0")
        self.assertEqual(payload["state_id"], "state:kv_cache")
        self.assertEqual(payload["version_label"], "before")
        back = gir.StateVersion.from_dict(payload)
        self.assertEqual(back, item)
        self.assertEqual((back.stable_id, back.state_id, back.version_label),
                         ("ver:kv_cache@t0", "state:kv_cache", "before"))

    def test_same_state_different_versions_are_distinct_facts(self):
        """同一 state_id 的两个版本仅 version_label 不同，仍是不同事实。"""
        before = state_version(version_label="before")
        after = state_version(stable_id="ver:kv_cache@t1", version_label="after")
        self.assertEqual(before.state_id, after.state_id)
        self.assertNotEqual(before, after)
        self.assertNotEqual(canonical(before), canonical(after))
        self.assertEqual(len({before, after}), 2)

    def test_same_label_different_state_ids_are_distinct_facts(self):
        a = state_version(state_id="state:kv_cache")
        b = state_version(stable_id="ver:kv_cache_b@t0", state_id="state:kv_cache_b")
        self.assertEqual(a.version_label, b.version_label)
        self.assertNotEqual(a, b)
        self.assertNotEqual(canonical(a), canonical(b))

    def test_same_explicit_stable_id_does_not_collapse_locally(self):
        """图级缺口（不算通过）：相同 stable_id 的版本本地均可构造且相等。"""
        first = state_version(stable_id="ver:dup")
        second = state_version(stable_id="ver:dup")
        self.assertEqual(first, second)
        self.assertEqual(len([first, second]), 2)

    def test_version_label_vocabulary_is_not_constrained(self):
        """边界观察：version_label 词表未约束，除空/空白外原样接受。"""
        for label in ("before", "after", "step-7", "0712T2030", "写作阶段", "v0"):
            with self.subTest(label=label):
                item = state_version(version_label=label)
                self.assertEqual(item.version_label, label)
                self.assertEqual(gir.StateVersion.from_dict(item.to_dict()), item)

    def test_required_text_fields_must_be_nonempty_strings(self):
        for field_name in STATE_VERSION_FIELDS:
            for bad in ("", "   ", "\t\n", None, 123, True, b"x", ["v"]):
                with self.subTest(field=field_name, bad=bad):
                    self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: state_version(**{f: b}), field_name
                    )

    def test_required_text_fields_reject_surrounding_whitespace(self):
        for field_name in STATE_VERSION_FIELDS:
            for bad in (" value", "value ", "\tvalue", "value\n"):
                with self.subTest(field=field_name, bad=bad):
                    exc = self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: state_version(**{f: b}), field_name
                    )
                    self.assertIn("空白", str(exc))

    def test_internal_whitespace_allowed(self):
        """边界观察：仅拒绝首尾空白，内部空格允许。"""
        item = state_version(stable_id="ver:kv cache@t0")
        self.assertEqual(item.stable_id, "ver:kv cache@t0")


# ---------------------------------------------------------------------------
# 3. StateVersion 往返、受控破坏与字节稳定
# ---------------------------------------------------------------------------
class StateVersionRoundTripTest(StateIdentityTestCase):
    def test_roundtrip_equality_and_dict_equality(self):
        item = state_version()
        back = gir.StateVersion.from_dict(item.to_dict())
        self.assertEqual(back, item)
        self.assertEqual(back.to_dict(), item.to_dict())

    def test_repeated_canonical_serialization_byte_identical(self):
        item = state_version()
        first = canonical(item)
        self.assertEqual(first.encode("utf-8"), canonical(item).encode("utf-8"))
        self.assertEqual(first, canonical(gir.StateVersion.from_dict(item.to_dict())))

    def test_canonical_bytes_deterministic_across_processes(self):
        item = state_version(version_label="写作阶段")
        expected = canonical(item)
        literal = repr(json.dumps(item.to_dict(), ensure_ascii=False))
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.graph_ir import StateVersion\n"
            f"data = json.loads({literal})\n"
            "v = StateVersion.from_dict(data)\n"
            "sys.stdout.write(json.dumps(v.to_dict(), ensure_ascii=False, sort_keys=True,"
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
        for bad in ([], "version", 3, None, True):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(lambda b=bad: gir.StateVersion.from_dict(b),
                                             "状态版本", "对象")

    def test_from_dict_missing_required_fields_named(self):
        for missing in STATE_VERSION_FIELDS:
            payload = state_version().to_dict()
            del payload[missing]
            with self.subTest(missing=missing):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.StateVersion.from_dict(p), missing
                )

    def test_from_dict_rejects_empty_and_whitespace_values(self):
        """反序列化路径同样逐字段校验（不是只查键存在）。"""
        for field_name in STATE_VERSION_FIELDS:
            for bad in ("", "   "):
                payload = state_version().to_dict()
                payload[field_name] = bad
                with self.subTest(field=field_name, bad=bad):
                    self.assertValueErrorLocated(
                        lambda p=payload: gir.StateVersion.from_dict(p), field_name
                    )

    def test_from_dict_extra_keys_ignored(self):
        """边界观察：未知多余键被忽略（前向兼容），不影响往返相等。"""
        payload = state_version().to_dict()
        payload["capacity"] = 4096
        payload["valid_length"] = 12
        self.assertEqual(gir.StateVersion.from_dict(payload), state_version())
        self.assertEqual(gir.StateVersion.from_dict(payload).to_dict(),
                         state_version().to_dict())


# ---------------------------------------------------------------------------
# 4. StateTransition 接口快照
# ---------------------------------------------------------------------------
class StateTransitionInterfaceTest(StateIdentityTestCase):
    def test_state_transition_is_frozen_dataclass_with_declared_field_order(self):
        self.assertTrue(gir.StateTransition.__dataclass_params__.frozen)
        self.assertEqual(
            tuple(f.name for f in fields(gir.StateTransition)),
            ("stable_id", "state_id", "before_version_id", "after_version_id",
             "evidence_refs"),
        )

    def test_required_fields_and_optional_defaults(self):
        by_name = {f.name: f for f in fields(gir.StateTransition)}
        for required in STATE_TRANSITION_REQUIRED:
            with self.subTest(required=required):
                self.assertIs(by_name[required].default, MISSING)
        self.assertEqual(by_name["evidence_refs"].default, ())
        minimal = gir.StateTransition(stable_id="t0", state_id="state:s",
                                      before_version_id="ver:a", after_version_id="ver:b")
        self.assertEqual(minimal.evidence_refs, ())

    def test_stable_id_is_explicit_field_not_derived_property(self):
        static = inspect.getattr_static(gir.StateTransition, "stable_id", None)
        self.assertNotIsInstance(static, property)
        item = state_transition(stable_id="plain-transition-id")
        self.assertEqual(item.stable_id, "plain-transition-id")
        self.assertNotEqual(item.stable_id, item.state_id)
        self.assertNotIn(item.stable_id, (item.before_version_id, item.after_version_id))

    def test_to_dict_has_exactly_five_keys(self):
        payload = state_transition().to_dict()
        self.assertEqual(set(payload), STATE_TRANSITION_TO_DICT_KEYS)
        self.assertEqual(len(payload), 5)

    def test_frozen_prevents_mutation_and_is_hashable(self):
        item = state_transition()
        for attribute, new_value in (("stable_id", "other"), ("state_id", "state:other"),
                                     ("before_version_id", "ver:x"),
                                     ("after_version_id", "ver:y"),
                                     ("evidence_refs", ())):
            with self.subTest(attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(item, attribute, new_value)
        self.assertEqual(hash(item), hash(state_transition()))
        self.assertEqual(len({item, state_transition()}), 1)

    def test_evidence_refs_frozen_to_tuple_with_defensive_copy(self):
        refs = ["fixture:state/ev1"]
        item = state_transition(evidence_refs=refs)
        self.assertIsInstance(item.evidence_refs, tuple)
        refs.append("fixture:state/ev3")
        self.assertEqual(item.evidence_refs, ("fixture:state/ev1",))

    def test_unexpected_constructor_keyword_rejected(self):
        with self.assertRaises(TypeError):
            gir.StateTransition(stable_id="t", state_id="s", before_version_id="a",
                                after_version_id="b", direction="forward")


# ---------------------------------------------------------------------------
# 5. StateTransition 方向保真：before != after、交换即不同事实
# ---------------------------------------------------------------------------
class StateTransitionDirectionTest(StateIdentityTestCase):
    def test_before_and_after_must_differ(self):
        exc = self.assertValueErrorLocated(
            lambda: state_transition(before_version_id="ver:same",
                                     after_version_id="ver:same"),
            "before_version_id", "after_version_id",
        )
        self.assertIn("ver:same", str(exc))  # 回显相同的值
        self.assertIn("不同", str(exc))

    def test_equal_before_after_also_detected_via_from_dict(self):
        payload = state_transition().to_dict()
        payload["after_version_id"] = payload["before_version_id"]
        self.assertValueErrorLocated(
            lambda: gir.StateTransition.from_dict(payload),
            "before_version_id", "after_version_id",
        )

    def test_direction_is_preserved_verbatim(self):
        item = state_transition()
        payload = item.to_dict()
        self.assertEqual(payload["before_version_id"], "ver:kv_cache@t0")
        self.assertEqual(payload["after_version_id"], "ver:kv_cache@t1")
        back = gir.StateTransition.from_dict(payload)
        self.assertEqual(back, item)
        self.assertEqual((back.before_version_id, back.after_version_id),
                         ("ver:kv_cache@t0", "ver:kv_cache@t1"))

    def test_swapping_before_and_after_yields_a_distinct_fact(self):
        forward = state_transition(before_version_id="ver:a", after_version_id="ver:b")
        backward = state_transition(before_version_id="ver:b", after_version_id="ver:a")
        self.assertNotEqual(forward, backward)
        self.assertNotEqual(canonical(forward), canonical(backward))
        self.assertEqual(forward.to_dict()["before_version_id"], "ver:a")
        self.assertEqual(backward.to_dict()["before_version_id"], "ver:b")

    def test_before_after_are_independent_of_stable_id_and_state_id(self):
        """边界观察：方向字段原样保存，不派生自 ID，也不校验版本前缀。"""
        item = state_transition(stable_id="t-arbitrary", state_id="state:anything",
                                before_version_id="opaque-a", after_version_id="opaque-b")
        self.assertEqual((item.before_version_id, item.after_version_id),
                         ("opaque-a", "opaque-b"))
        self.assertEqual(gir.StateTransition.from_dict(item.to_dict()), item)

    def test_before_after_required_text_rules(self):
        for field_name in ("before_version_id", "after_version_id"):
            for bad in ("", "   ", "\t", None, 5, True, b"v"):
                with self.subTest(field=field_name, bad=bad):
                    self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: state_transition(**{f: b}), field_name
                    )
            for bad in (" ver:x", "ver:x ", "\tver:x"):
                with self.subTest(field=field_name, whitespace=bad):
                    exc = self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: state_transition(**{f: b}), field_name
                    )
                    self.assertIn("空白", str(exc))

    def test_state_id_and_stable_id_required_text_rules(self):
        for field_name in ("stable_id", "state_id"):
            for bad in ("", "   ", None, 5, True):
                with self.subTest(field=field_name, bad=bad):
                    self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: state_transition(**{f: b}), field_name
                    )
            for bad in (" value", "value "):
                with self.subTest(field=field_name, whitespace=bad):
                    exc = self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: state_transition(**{f: b}), field_name
                    )
                    self.assertIn("空白", str(exc))

    def test_reverse_direction_reference_is_locally_representable(self):
        """图级缺口（不算通过）：反向/任意版本引用本地均可构造。"""
        item = state_transition(before_version_id="ver:kv_cache@t9",
                                after_version_id="ver:kv_cache@t0")
        self.assertEqual((item.before_version_id, item.after_version_id),
                         ("ver:kv_cache@t9", "ver:kv_cache@t0"))
        self.assertEqual(gir.StateTransition.from_dict(item.to_dict()), item)


# ---------------------------------------------------------------------------
# 6. 有序 evidence_refs
# ---------------------------------------------------------------------------
class StateTransitionEvidenceRefsTest(StateIdentityTestCase):
    def test_default_empty_and_roundtrip(self):
        item = state_transition(evidence_refs=[])
        self.assertEqual(item.evidence_refs, ())
        self.assertEqual(item.to_dict()["evidence_refs"], [])
        self.assertEqual(gir.StateTransition.from_dict(item.to_dict()).evidence_refs, ())

    def test_order_preserved(self):
        refs = ["fixture:state/a", "fixture:state/b", "fixture:state/c"]
        item = state_transition(evidence_refs=refs)
        self.assertEqual(list(item.evidence_refs), refs)
        self.assertEqual(item.to_dict()["evidence_refs"], refs)
        self.assertEqual(list(gir.StateTransition.from_dict(item.to_dict()).evidence_refs),
                         refs)
        reversed_item = state_transition(evidence_refs=list(reversed(refs)))
        self.assertNotEqual(canonical(item), canonical(reversed_item))

    def test_items_must_be_nonempty_strings_with_index(self):
        for bad, needle in (([None], "evidence_refs[0]"), ([""], "evidence_refs[0]"),
                            (["   "], "evidence_refs[0]"), ([5], "evidence_refs[0]"),
                            ([[]], "evidence_refs[0]"),
                            (["ok", None], "evidence_refs[1]")):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: state_transition(evidence_refs=b), needle
                )

    def test_container_must_be_list_or_tuple(self):
        for bad in ({"fixture:a"}, {"fixture:a": 1}, (x for x in ["fixture:a"])):
            with self.subTest(bad=type(bad).__name__):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: state_transition(evidence_refs=b), "evidence_refs"
                )
                self.assertIn("list/tuple", str(exc))
                self.assertIn(type(bad).__name__, str(exc))

    def test_string_container_is_not_split_into_characters(self):
        """受控反例：字符串不是序列容器，应整串拒绝而非拆成单字符列表。"""
        exc = self.assertValueErrorLocated(
            lambda: state_transition(evidence_refs="fixture:state/ev1"), "evidence_refs"
        )
        self.assertIn("list/tuple", str(exc))
        self.assertIn("str", str(exc))

    def test_dict_container_keys_are_not_taken(self):
        exc = self.assertValueErrorLocated(
            lambda: state_transition(evidence_refs={"fixture:state/ev1": 1}),
            "evidence_refs",
        )
        self.assertIn("dict", str(exc))

    def test_list_and_tuple_accepted(self):
        self.assertEqual(state_transition(evidence_refs=["fixture:a"]).evidence_refs,
                         ("fixture:a",))
        self.assertEqual(state_transition(evidence_refs=("fixture:a",)).evidence_refs,
                         ("fixture:a",))

    def test_duplicates_allowed(self):
        """边界观察：未规定引用去重，重复项原样保留。"""
        item = state_transition(evidence_refs=["fixture:same", "fixture:same"])
        self.assertEqual(item.evidence_refs, ("fixture:same", "fixture:same"))
        self.assertEqual(gir.StateTransition.from_dict(item.to_dict()).evidence_refs,
                         ("fixture:same", "fixture:same"))

    def test_evidence_refs_not_required_to_be_nonempty_list(self):
        """边界观察：evidence_refs 可选，不强制至少一条证据。"""
        item = state_transition(evidence_refs=())
        self.assertEqual(item.evidence_refs, ())

    def test_evidence_refs_do_not_collapse_distinct_transitions(self):
        a = state_transition(stable_id="t_a", evidence_refs=["fixture:1"])
        b = state_transition(stable_id="t_b", evidence_refs=["fixture:1"])
        self.assertNotEqual(a, b)


# ---------------------------------------------------------------------------
# 7. StateTransition 往返、JSON 数组边界与字节稳定
# ---------------------------------------------------------------------------
class StateTransitionRoundTripTest(StateIdentityTestCase):
    def test_roundtrip_equality(self):
        item = state_transition()
        back = gir.StateTransition.from_dict(item.to_dict())
        self.assertEqual(back, item)
        self.assertEqual(back.to_dict(), item.to_dict())
        self.assertEqual(canonical(back), canonical(item))

    def test_repeated_canonical_serialization_byte_identical(self):
        item = state_transition()
        first = canonical(item)
        self.assertEqual(first.encode("utf-8"), canonical(item).encode("utf-8"))
        self.assertEqual(first, canonical(gir.StateTransition.from_dict(item.to_dict())))

    def test_canonical_bytes_deterministic_across_processes(self):
        item = state_transition(evidence_refs=["fixture:state/写作", "fixture:state/ev2"])
        expected = canonical(item)
        literal = repr(json.dumps(item.to_dict(), ensure_ascii=False))
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.graph_ir import StateTransition\n"
            f"data = json.loads({literal})\n"
            "t = StateTransition.from_dict(data)\n"
            "sys.stdout.write(json.dumps(t.to_dict(), ensure_ascii=False, sort_keys=True,"
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
        for bad in ([], "transition", 3, None, True):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: gir.StateTransition.from_dict(b), "状态更新", "对象"
                )

    def test_from_dict_missing_required_fields_named(self):
        for missing in STATE_TRANSITION_REQUIRED:
            payload = state_transition().to_dict()
            del payload[missing]
            with self.subTest(missing=missing):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.StateTransition.from_dict(p), missing
                )

    def test_from_dict_evidence_refs_must_be_json_array(self):
        for bad in (None, 5, True, "ev1", "", {"fixture:a": 1}):
            payload = state_transition().to_dict()
            payload["evidence_refs"] = bad
            with self.subTest(bad=bad):
                exc = self.assertValueErrorLocated(
                    lambda p=payload: gir.StateTransition.from_dict(p), "evidence_refs"
                )
                self.assertIn("JSON 数组", str(exc))

    def test_from_dict_optional_evidence_refs_absent_default(self):
        payload = {"stable_id": "t0", "state_id": "state:s",
                   "before_version_id": "ver:a", "after_version_id": "ver:b"}
        item = gir.StateTransition.from_dict(payload)
        self.assertEqual(item.evidence_refs, ())
        self.assertEqual(item, gir.StateTransition(stable_id="t0", state_id="state:s",
                                                  before_version_id="ver:a",
                                                  after_version_id="ver:b"))

    def test_from_dict_extra_keys_ignored(self):
        payload = state_transition().to_dict()
        payload["capacity"] = 4096
        payload["valid_length"] = 12
        self.assertEqual(gir.StateTransition.from_dict(payload), state_transition())
        self.assertEqual(gir.StateTransition.from_dict(payload).to_dict(),
                         state_transition().to_dict())

    def test_distinct_transitions_do_not_collapse(self):
        a = state_transition(stable_id="t_a")
        b = state_transition(stable_id="t_b")
        c = state_transition(stable_id="t_a", after_version_id="ver:kv_cache@t2")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len({a, b, c}), 3)


# ---------------------------------------------------------------------------
# 8. Edge ↔ StateTransition 引用约定（state 边 vs 普通边）
# ---------------------------------------------------------------------------
class EdgeStateTransitionReferenceTest(StateIdentityTestCase):
    @staticmethod
    def _state_edge(**overrides):
        spec = {
            "stable_id": "edge:kv_state@layer1",
            "src_node_id": "call:attn@layer0",
            "src_port": "kv_out",
            "dst_node_id": "call:attn@layer1",
            "dst_port": "kv_in",
            "tensor_id": "tensor:kv",
            "kind": "state",
        }
        spec.update(overrides)
        return gir.Edge(**spec)

    @staticmethod
    def _data_edge(**overrides):
        spec = {
            "stable_id": "edge:q_out->attn_in@layer0",
            "src_node_id": "call:q_proj@layer0",
            "src_port": "q_out",
            "dst_node_id": "call:attn@layer0",
            "dst_port": "q_in",
            "tensor_id": "tensor:q",
            "kind": "data",
        }
        spec.update(overrides)
        return gir.Edge(**spec)

    def test_state_edge_can_reference_transition_stable_id(self):
        transition = state_transition()
        item = self._state_edge(state_transition_ref=transition.stable_id)
        self.assertEqual(item.state_transition_ref, transition.stable_id)
        self.assertEqual(item.kind, "state")
        back = gir.Edge.from_dict(item.to_dict())
        self.assertEqual(back, item)
        self.assertEqual(back.state_transition_ref, transition.stable_id)

    def test_reference_is_an_opaque_string_copy(self):
        """引用是字符串（值相等即可），不是对象引用；本地不解析到对象。"""
        transition = state_transition()
        item = self._state_edge(state_transition_ref="".join([transition.stable_id]))
        self.assertEqual(item.state_transition_ref, transition.stable_id)
        self.assertIsInstance(item.state_transition_ref, str)

    def test_state_edge_without_reference_still_rejected(self):
        """Edge 既有双向规则未被状态类新增破坏。"""
        self.assertValueErrorLocated(
            lambda: self._state_edge(state_transition_ref=None), "state_transition_ref"
        )

    def test_non_state_edges_still_forbid_reference(self):
        transition = state_transition()
        for kind in ("data", "select", "control"):
            with self.subTest(kind=kind):
                exc = self.assertValueErrorLocated(
                    lambda k=kind: self._data_edge(
                        kind=k, state_transition_ref=transition.stable_id),
                    "state_transition_ref",
                )
                self.assertIn(kind, str(exc))

    def test_non_state_edges_still_forbid_reference_via_from_dict(self):
        transition = state_transition()
        for kind in ("data", "select", "control"):
            payload = self._data_edge(kind=kind).to_dict()
            payload["state_transition_ref"] = transition.stable_id
            with self.subTest(kind=kind):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.Edge.from_dict(p), "state_transition_ref"
                )

    def test_reference_to_nonexistent_transition_is_locally_accepted(self):
        """图级缺口（不算通过）：引用存在性本地不检查。"""
        item = self._state_edge(state_transition_ref="trans:does_not_exist")
        self.assertEqual(item.state_transition_ref, "trans:does_not_exist")
        self.assertEqual(gir.Edge.from_dict(item.to_dict()), item)

    def test_transition_does_not_require_referenced_versions_to_exist(self):
        """图级缺口（不算通过）：before/after 指向的版本是否存在本地不检查。"""
        item = state_transition(before_version_id="ver:missing_a",
                                after_version_id="ver:missing_b")
        self.assertEqual(gir.StateTransition.from_dict(item.to_dict()), item)


# ---------------------------------------------------------------------------
# 9. cache 属性彻底未实现 + 图级缺口（不算通过）
# ---------------------------------------------------------------------------
class UnimplementedCacheAttributesTest(StateIdentityTestCase):
    def test_state_models_carry_no_cache_numeric_attributes(self):
        """任务单要求明确：capacity/valid_length/remainder/overlap/Indexer 状态尚不存在。

        本单**不是完整 cache 测试**；这里固定"当前不存在"这一事实，而不是通过某个 cache 契约。
        """
        version_fields = {f.name for f in fields(gir.StateVersion)}
        transition_fields = {f.name for f in fields(gir.StateTransition)}
        for name in UNIMPLEMENTED_CACHE_FIELDS:
            with self.subTest(attribute=name):
                self.assertNotIn(name, version_fields)
                self.assertNotIn(name, transition_fields)
                self.assertFalse(hasattr(state_version(), name))
                self.assertFalse(hasattr(state_transition(), name))
        self.assertEqual(version_fields, set(STATE_VERSION_FIELDS))
        self.assertEqual(transition_fields, set(STATE_TRANSITION_TO_DICT_KEYS))

    def test_no_compression_or_indexer_concept_in_serialized_form(self):
        for item in (state_version(), state_transition()):
            payload = item.to_dict()
            for name in UNIMPLEMENTED_CACHE_FIELDS:
                with self.subTest(cls=type(item).__name__, attribute=name):
                    self.assertNotIn(name, payload)

    def test_transition_state_id_consistency_with_versions_not_checked(self):
        """图级缺口：transition.state_id 与所引用版本所属 state_id 的一致性本地不查。"""
        before = state_version(state_id="state:kv_cache")
        after = state_version(stable_id="ver:other@t0", state_id="state:other_cache",
                              version_label="after")
        item = state_transition(state_id="state:third_cache",
                                before_version_id=before.stable_id,
                                after_version_id=after.stable_id)
        self.assertNotEqual(before.state_id, after.state_id)
        self.assertNotEqual(item.state_id, before.state_id)
        self.assertEqual(gir.StateTransition.from_dict(item.to_dict()), item)

    def test_no_version_cycle_or_ordering_check(self):
        """图级缺口：无版本环/跨调用先后本地不检查。"""
        forward = state_transition(stable_id="t1", before_version_id="ver:a",
                                   after_version_id="ver:b")
        backward = state_transition(stable_id="t2", before_version_id="ver:b",
                                    after_version_id="ver:a")
        self.assertEqual((forward.before_version_id, forward.after_version_id),
                         ("ver:a", "ver:b"))
        self.assertEqual((backward.before_version_id, backward.after_version_id),
                         ("ver:b", "ver:a"))
        self.assertNotEqual(forward, backward)

    def test_graph_level_stable_id_uniqueness_not_enforced(self):
        first = state_transition(stable_id="t_dup")
        second = state_transition(stable_id="t_dup")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
