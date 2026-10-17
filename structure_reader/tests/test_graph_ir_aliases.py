"""S1 01a AliasRelation 独立验证：共享同张量、view/slice 派生、按 kind 分叉的局部不变量。

范围严格限定于 `results/development_01a_alias_probe.md` 声明的增量：
- `AliasRelation`：显式 `stable_id`（字段，非派生 property）、`base_tensor_id`、
  `alias_tensor_id`、`relation_kind` 四必填字符串 + 可选 `alias_role`、`transform_expr`、
  有序可选 `evidence_refs`；`to_dict` 恰 7 键；`from_dict` 拒绝非对象/缺字段/
  `evidence_refs` 非 JSON 数组；frozen、可哈希、重复与跨进程字节稳定。
- `relation_kind` 限 `same_tensor|view|slice`（大小写敏感，非法值回显值），**局部不变量按 kind 分叉**：
  - `same_tensor`：两 tensor ID **必须相同**、`alias_role` 必填非空、`transform_expr` 必须为 None；
  - `view`/`slice`：两 tensor ID **必须不同**、`transform_expr` 必填非空（**原文只存储、不执行**）、
    `alias_role` 必须为 None。
- 目标表达（不得由 alias 生成新张量/投影/权重事实）：K/V 两个消费端口**共同引用同一个 tensor ID**，
  两条 `same_tensor` 关系仅以 `alias_role` 区分角色；view/slice 只说派生关系，不声称独立存储。

**明确属图级/源码检查、本单不通过也不声称**（见 `AliasGraphLevelGapsTest`）：
base/alias 张量**引用存在性**、同一角色**重复绑定**、存储别名与**源码证据/物理内存别名真实性**、
view/slice 的**轴变换相容性**、图级 `stable_id` 唯一性、`transform_expr` 求值。

不在本范围：Guard/GuardActivation、状态身份、SemanticGraph、`graph_to_json`/`graph_from_json`、
图级 canonical 序列化、01b 检查器、01c schema、有限求值器（开发单 02）、源码核验（开发单 03）、
真实 DeepSeek 结构提取。

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
import tempfile
import unittest
from dataclasses import FrozenInstanceError, MISSING, fields
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import structure_reader.graph_ir as gir  # noqa: E402

ALIAS_REQUIRED = ("stable_id", "base_tensor_id", "alias_tensor_id", "relation_kind")
ALIAS_TO_DICT_KEYS = {"stable_id", "base_tensor_id", "alias_tensor_id", "relation_kind",
                      "alias_role", "transform_expr", "evidence_refs"}
ALIAS_KINDS = ("same_tensor", "view", "slice")
CANONICAL_KWARGS = {"ensure_ascii": False, "sort_keys": True, "separators": (",", ":")}
SHARED_TENSOR_ID = "tensor:kv_proj_out"


def alias_same(**overrides):
    """合成契约样例：共享 KV 的 key 角色（两个 ID 相同，无 transform）。"""
    spec = {
        "stable_id": "alias:kv@layer0#key",
        "base_tensor_id": SHARED_TENSOR_ID,
        "alias_tensor_id": SHARED_TENSOR_ID,
        "relation_kind": "same_tensor",
        "alias_role": "key",
        "evidence_refs": ["fixture:alias/ev1", "fixture:alias/ev2"],
    }
    spec.update(overrides)
    return gir.AliasRelation(**spec)


def alias_view(**overrides):
    """合成契约样例：缓存 view（base/alias 不同 + 变换原文）。"""
    spec = {
        "stable_id": "alias:k_cache_view@layer0",
        "base_tensor_id": "tensor:k_cache_full",
        "alias_tensor_id": "tensor:k_cache_window",
        "relation_kind": "view",
        "transform_expr": "k_cache_full[..., -W:, :]",
    }
    spec.update(overrides)
    return gir.AliasRelation(**spec)


def alias_slice(**overrides):
    """合成契约样例：缓存 slice（base/alias 不同 + 变换原文）。"""
    spec = {
        "stable_id": "alias:k_cache_slice@layer0",
        "base_tensor_id": "tensor:k_cache_full",
        "alias_tensor_id": "tensor:k_cache_slice",
        "relation_kind": "slice",
        "transform_expr": "k_cache_full[:, :, -W:, :]",
    }
    spec.update(overrides)
    return gir.AliasRelation(**spec)


def canonical(item):
    """仅用于稳定性比较：对 to_dict() 做确定性 JSON 编码（非产品 API）。"""
    return json.dumps(item.to_dict(), **CANONICAL_KWARGS)


class AliasTestCase(unittest.TestCase):
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
class AliasInterfaceTest(AliasTestCase):
    def test_alias_is_frozen_dataclass_with_declared_field_order(self):
        self.assertTrue(gir.AliasRelation.__dataclass_params__.frozen)
        self.assertEqual(
            tuple(f.name for f in fields(gir.AliasRelation)),
            ("stable_id", "base_tensor_id", "alias_tensor_id", "relation_kind",
             "alias_role", "transform_expr", "evidence_refs"),
        )

    def test_required_fields_and_optional_defaults(self):
        by_name = {f.name: f for f in fields(gir.AliasRelation)}
        for required in ALIAS_REQUIRED:
            with self.subTest(required=required):
                self.assertIs(by_name[required].default, MISSING)
        self.assertIsNone(by_name["alias_role"].default)
        self.assertIsNone(by_name["transform_expr"].default)
        self.assertEqual(by_name["evidence_refs"].default, ())
        minimal = gir.AliasRelation(stable_id="a0", base_tensor_id=SHARED_TENSOR_ID,
                                    alias_tensor_id=SHARED_TENSOR_ID,
                                    relation_kind="same_tensor", alias_role="key")
        self.assertIsNone(minimal.transform_expr)
        self.assertEqual(minimal.evidence_refs, ())

    def test_stable_id_is_explicit_field_not_derived_property(self):
        static = inspect.getattr_static(gir.AliasRelation, "stable_id", None)
        self.assertNotIsInstance(static, property)
        item = alias_same(stable_id="plain-alias-id")
        self.assertEqual(item.stable_id, "plain-alias-id")

    def test_to_dict_has_exactly_seven_keys(self):
        for item in (alias_same(), alias_view(), alias_slice()):
            with self.subTest(kind=item.relation_kind):
                payload = item.to_dict()
                self.assertEqual(set(payload), ALIAS_TO_DICT_KEYS)
                self.assertEqual(len(payload), 7)

    def test_frozen_prevents_mutation_and_is_hashable(self):
        item = alias_same()
        for attribute, new_value in (("stable_id", "other"),
                                     ("relation_kind", "view"),
                                     ("alias_role", None),
                                     ("evidence_refs", ())):
            with self.subTest(attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(item, attribute, new_value)
        self.assertEqual(hash(item), hash(alias_same()))
        self.assertEqual(len({item, alias_same()}), 1)

    def test_unexpected_constructor_keyword_rejected(self):
        with self.assertRaises(TypeError):
            gir.AliasRelation(stable_id="a", base_tensor_id="t", alias_tensor_id="t",
                              relation_kind="same_tensor", alias_role="key",
                              tensor_id="tensor:t")

    def test_no_field_fabricates_a_new_tensor_projection_or_weight(self):
        """不得由 alias 生成两个 tensor/投影/权重事实：字段集里没有任何"新张量"字段。"""
        field_names = {f.name for f in fields(gir.AliasRelation)}
        for forbidden in ("new_tensor_id", "created_tensor_id", "projection",
                          "projection_id", "weight", "weight_id", "tensor", "storage"):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, field_names)
        self.assertEqual(field_names, ALIAS_TO_DICT_KEYS)


# ---------------------------------------------------------------------------
# 2. relation_kind：三种取值、大小写敏感、非法值
# ---------------------------------------------------------------------------
class AliasKindTest(AliasTestCase):
    def test_three_kinds_accepted_with_their_own_contracts(self):
        items = (alias_same(), alias_view(), alias_slice())
        self.assertEqual({item.relation_kind for item in items}, set(ALIAS_KINDS))
        for item in items:
            with self.subTest(kind=item.relation_kind):
                self.assertEqual(gir.AliasRelation.from_dict(item.to_dict()), item)

    def test_kind_is_case_sensitive(self):
        for bad in ("SameTensor", "SAMETENSOR", "View", "VIEW", "Slice", "SLICE"):
            with self.subTest(kind=bad):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: gir.AliasRelation(
                        stable_id="a", base_tensor_id="t", alias_tensor_id="t",
                        relation_kind=b, alias_role="key"),
                    "relation_kind",
                )
                self.assertIn(repr(bad), str(exc))

    def test_kind_surrounding_whitespace_rejected_before_whitelist(self):
        for bad in ("same_tensor ", " view", "\tslice"):
            with self.subTest(kind=bad):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: gir.AliasRelation(
                        stable_id="a", base_tensor_id="t1", alias_tensor_id="t2",
                        relation_kind=b, transform_expr="x"),
                    "relation_kind",
                )
                self.assertIn("空白", str(exc))

    def test_invalid_kind_values_rejected_with_field_name(self):
        for bad in ("", "   ", "alias", None, 5, True, ["same_tensor"],
                    {"relation_kind": "view"}):
            with self.subTest(kind=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: alias_same(relation_kind=b), "relation_kind"
                )

    def test_kind_error_lists_all_three_legal_values(self):
        exc = self.assertValueErrorLocated(
            lambda: alias_same(relation_kind="copy"), "relation_kind"
        )
        for kind in ALIAS_KINDS:
            self.assertIn(kind, str(exc))
        self.assertIn("copy", str(exc))

    def test_kind_whitelist_precedes_kind_specific_branch(self):
        """kind 非法 + 同 ID 冲突时，错误应指向 relation_kind（先校验白名单）。"""
        exc = self.assertValueErrorLocated(
            lambda: gir.AliasRelation(stable_id="a", base_tensor_id="t1",
                                      alias_tensor_id="t2", relation_kind="bogus",
                                      alias_role="key", transform_expr="x"),
            "relation_kind",
        )
        self.assertIn("bogus", str(exc))

    def test_required_text_rule_precedes_kind_whitelist(self):
        exc = self.assertValueErrorLocated(
            lambda: gir.AliasRelation(stable_id="", base_tensor_id="t1",
                                      alias_tensor_id="t2", relation_kind="bogus"),
            "stable_id",
        )
        self.assertIn("stable_id", str(exc))

    def test_each_kind_is_a_distinct_fact(self):
        self.assertEqual(len({canonical(alias_same()), canonical(alias_view()),
                              canonical(alias_slice())}), 3)

    def test_kind_specific_messages_name_the_kind(self):
        exc_same = self.assertValueErrorLocated(
            lambda: alias_same(base_tensor_id="tensor:a", alias_tensor_id="tensor:b"),
            "same_tensor")
        self.assertIn("same_tensor", str(exc_same))
        exc_view = self.assertValueErrorLocated(
            lambda: alias_view(base_tensor_id="tensor:a", alias_tensor_id="tensor:a"),
            "view")
        self.assertIn("view", str(exc_view))
        exc_slice = self.assertValueErrorLocated(
            lambda: alias_slice(base_tensor_id="tensor:a", alias_tensor_id="tensor:a"),
            "slice")
        self.assertIn("slice", str(exc_slice))


# ---------------------------------------------------------------------------
# 3. same_tensor：同 ID、角色必填、不得造新张量
# ---------------------------------------------------------------------------
class AliasSameTensorTest(AliasTestCase):
    def test_same_tensor_with_equal_ids_and_role_accepted_and_roundtrip(self):
        item = alias_same()
        self.assertEqual(item.base_tensor_id, item.alias_tensor_id)
        self.assertEqual(item.alias_role, "key")
        self.assertIsNone(item.transform_expr)
        payload = item.to_dict()
        self.assertEqual(payload["base_tensor_id"], payload["alias_tensor_id"])
        self.assertIsNone(payload["transform_expr"])
        self.assertEqual(gir.AliasRelation.from_dict(payload), item)

    def test_same_tensor_requires_identical_tensor_ids(self):
        exc = self.assertValueErrorLocated(
            lambda: alias_same(base_tensor_id="tensor:k", alias_tensor_id="tensor:v"),
            "base_tensor_id", "alias_tensor_id",
        )
        self.assertIn("tensor:k", str(exc))
        self.assertIn("tensor:v", str(exc))

    def test_same_tensor_requires_alias_role(self):
        exc = self.assertValueErrorLocated(
            lambda: alias_same(alias_role=None), "alias_role"
        )
        self.assertIn("same_tensor", str(exc))

    def test_same_tensor_role_must_be_nonempty_string(self):
        for bad in ("", "   ", "\t", 5, True, ["key"]):
            with self.subTest(role=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: alias_same(alias_role=b), "alias_role"
                )

    def test_same_tensor_role_rejects_surrounding_whitespace(self):
        for bad in (" key", "key ", "\tkey"):
            with self.subTest(role=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: alias_same(alias_role=b), "alias_role"
                )

    def test_same_tensor_forbids_transform_expr(self):
        for bad in ("x", "", "   ", 5):
            with self.subTest(transform=bad):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: alias_same(transform_expr=b), "transform_expr"
                )
                self.assertIn("transform_expr", str(exc))

    def test_same_tensor_forbids_transform_expr_even_when_nonempty(self):
        exc = self.assertValueErrorLocated(
            lambda: alias_same(transform_expr="anything"), "same_tensor", "transform_expr"
        )
        self.assertIn("transform_expr", str(exc))

    def test_key_and_value_roles_share_one_tensor_id(self):
        """共享 KV：key/value 两条 same_tensor 关系指向同一个 tensor ID，角色不同。"""
        key_rel = alias_same(stable_id="alias:kv#key", alias_role="key")
        value_rel = alias_same(stable_id="alias:kv#value", alias_role="value")
        self.assertEqual(key_rel.base_tensor_id, value_rel.base_tensor_id)
        self.assertEqual(key_rel.alias_tensor_id, value_rel.alias_tensor_id)
        self.assertEqual(key_rel.base_tensor_id, key_rel.alias_tensor_id)
        self.assertNotEqual(key_rel.alias_role, value_rel.alias_role)
        self.assertNotEqual(key_rel, value_rel)
        self.assertNotEqual(canonical(key_rel), canonical(value_rel))
        distinct_ids = {key_rel.base_tensor_id, key_rel.alias_tensor_id,
                        value_rel.base_tensor_id, value_rel.alias_tensor_id}
        self.assertEqual(distinct_ids, {SHARED_TENSOR_ID})

    def test_two_consumption_ports_reference_the_same_tensor_id(self):
        """构造两个实际消费端口共同引用该 tensor ID：alias 不派生第二个张量事实。"""
        port_k = gir.Port(name="k_in", direction="input", tensor_id=SHARED_TENSOR_ID)
        port_v = gir.Port(name="v_in", direction="input", tensor_id=SHARED_TENSOR_ID)
        node = gir.Node(stable_id="call:attn@layer0", op_type="attn",
                        granularity="op", input_ports=[port_k, port_v])
        key_rel = alias_same(stable_id="alias:kv#key", alias_role="key")
        value_rel = alias_same(stable_id="alias:kv#value", alias_role="value")
        referenced = {port.tensor_id for port in node.input_ports}
        self.assertEqual(referenced, {SHARED_TENSOR_ID})
        self.assertEqual({key_rel.base_tensor_id, value_rel.base_tensor_id}, referenced)
        self.assertEqual({key_rel.alias_tensor_id, value_rel.alias_tensor_id}, referenced)
        # 端口的张量 ID 集合与 alias 引用集合都只含同一个 ID，没有第二个张量身份
        self.assertEqual(len(referenced | {key_rel.base_tensor_id}), 1)
        # alias 关系本身不是张量：它不是 tensor: 前缀（仅为不透明 ID 的观察，不作格式强约束）
        self.assertNotIn("tensor", key_rel.stable_id)

    def test_roundtrip_keeps_single_tensor_identity(self):
        key_rel = alias_same(alias_role="key")
        back = gir.AliasRelation.from_dict(key_rel.to_dict())
        self.assertEqual(back.base_tensor_id, back.alias_tensor_id)
        self.assertIsNone(back.transform_expr)

    def test_same_role_on_different_tensor_ids_are_locally_permitted(self):
        """图级缺口（不算通过）：同一 alias_role 绑到其它 tensor ID 本地不检查。"""
        a = alias_same(stable_id="a1", alias_role="key",
                       base_tensor_id="tensor:x", alias_tensor_id="tensor:x")
        b = alias_same(stable_id="a2", alias_role="key",
                       base_tensor_id="tensor:y", alias_tensor_id="tensor:y")
        self.assertEqual(a.alias_role, b.alias_role)
        self.assertNotEqual(a.base_tensor_id, b.base_tensor_id)
        self.assertNotEqual(a, b)


# ---------------------------------------------------------------------------
# 4. view / slice：不同 ID + 变换原文（只存储不执行）
# ---------------------------------------------------------------------------
class AliasViewSliceTest(AliasTestCase):
    def test_view_and_slice_accepted_and_roundtrip(self):
        for item in (alias_view(), alias_slice()):
            with self.subTest(kind=item.relation_kind):
                self.assertNotEqual(item.base_tensor_id, item.alias_tensor_id)
                self.assertIsInstance(item.transform_expr, str)
                self.assertTrue(item.transform_expr.strip())
                self.assertIsNone(item.alias_role)
                back = gir.AliasRelation.from_dict(item.to_dict())
                self.assertEqual(back, item)
                self.assertEqual(back.transform_expr, item.transform_expr)

    def test_view_slice_require_different_tensor_ids(self):
        for kind, factory in (("view", alias_view), ("slice", alias_slice)):
            with self.subTest(kind=kind):
                exc = self.assertValueErrorLocated(
                    lambda f=factory: f(base_tensor_id="tensor:same",
                                        alias_tensor_id="tensor:same"),
                    "base_tensor_id", "alias_tensor_id",
                )
                self.assertIn(kind, str(exc))
                self.assertIn("tensor:same", str(exc))

    def test_view_slice_require_transform_expr(self):
        for kind, factory in (("view", alias_view), ("slice", alias_slice)):
            with self.subTest(kind=kind):
                exc = self.assertValueErrorLocated(
                    lambda f=factory: f(transform_expr=None), "transform_expr"
                )
                self.assertIn(kind, str(exc))

    def test_view_slice_transform_expr_must_be_nonempty_string(self):
        for kind, factory in (("view", alias_view), ("slice", alias_slice)):
            for bad in ("", "   ", "\t", 5, True, ["expr"]):
                with self.subTest(kind=kind, bad=bad):
                    self.assertValueErrorLocated(
                        lambda f=factory, b=bad: f(transform_expr=b), "transform_expr"
                    )

    def test_view_slice_transform_expr_rejects_surrounding_whitespace(self):
        for bad in (" expr", "expr ", "\texpr"):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: alias_view(transform_expr=b), "transform_expr"
                )

    def test_view_slice_forbid_alias_role(self):
        for kind, factory in (("view", alias_view), ("slice", alias_slice)):
            for bad in ("key", "", "   ", 5):
                with self.subTest(kind=kind, role=bad):
                    exc = self.assertValueErrorLocated(
                        lambda f=factory, b=bad: f(alias_role=b), "alias_role"
                    )
                    self.assertIn("alias_role", str(exc))

    def test_transform_expr_is_preserved_verbatim(self):
        for expr in ("k_cache_full[..., -W:, :]", "x.reshape(B, H, S, D)",
                     "cache[:, :, start:start+W, :]", "torch.narrow(x, 2, 0, W)",
                     "切片[..., -W:, :]"):
            with self.subTest(expr=expr):
                item = alias_view(transform_expr=expr)
                self.assertEqual(item.transform_expr, expr)
                self.assertEqual(gir.AliasRelation.from_dict(item.to_dict()).transform_expr,
                                 expr)

    def test_multi_line_transform_expr_allowed_but_surrounding_whitespace_rejected(self):
        """边界观察：变换原文允许多行；仅拒绝首尾空白。

        首尾空白由通用可选字段规则拒绝，错误信息为
        `transform_expr 必须是非空字符串或 None`（不含"空白"字样），故只断言字段级定位。
        """
        item = alias_view(transform_expr="cache[:, -W:, :]\n  .contiguous()")
        self.assertEqual(item.transform_expr, "cache[:, -W:, :]\n  .contiguous()")
        exc = self.assertValueErrorLocated(
            lambda: alias_view(transform_expr="\nexpr"), "transform_expr"
        )
        self.assertIn("transform_expr", str(exc))

    def test_transform_expr_is_never_executed(self):
        """只存储不执行：把有副作用的表达式放进 transform_expr，构造后不得有副作用。"""
        marker = Path(tempfile.gettempdir()) / f"dsh_alias_transform_{os.getpid()}.marker"
        if marker.exists():
            marker.unlink()
        try:
            expr = f"__import__('pathlib').Path({str(marker)!r}).write_text('pwned') or 'x'"
            item = alias_view(transform_expr=expr)
            self.assertEqual(item.transform_expr, expr)
            self.assertEqual(gir.AliasRelation.from_dict(item.to_dict()).transform_expr, expr)
            self.assertFalse(marker.exists(), "transform_expr 被求值/执行了（出现副作用）")
        finally:
            if marker.exists():
                marker.unlink()

    def test_alias_section_source_contains_no_dynamic_execution(self):
        """静态佐证：AliasRelation 段源码不含 eval/exec/__import__/compile。"""
        source = (SRC_DIR / "structure_reader" / "graph_ir.py").read_text(encoding="utf-8")
        start = source.index("class AliasRelation:")
        end = source.index("def _require_unique_symbols")
        section = source[start:end]
        for token in ("eval(", "exec(", "__import__", "compile("):
            with self.subTest(token=token):
                self.assertNotIn(token, section)

    def test_importing_product_does_not_import_model_runtimes(self):
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

    def test_view_and_slice_are_distinct_facts(self):
        self.assertNotEqual(alias_view(), alias_slice())
        self.assertNotEqual(canonical(alias_view()), canonical(alias_slice()))
        same_ids_view = alias_view(base_tensor_id="tensor:b", alias_tensor_id="tensor:a",
                                   stable_id="alias:v")
        self.assertNotEqual(canonical(same_ids_view), canonical(alias_view()))

    def test_view_does_not_claim_independent_storage(self):
        payload = alias_view().to_dict()
        for forbidden in ("independent_storage", "owns_storage", "allocated", "size_bytes"):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, payload)
        self.assertNotEqual(payload["base_tensor_id"], payload["alias_tensor_id"])


# ---------------------------------------------------------------------------
# 5. 有序 evidence_refs（两道闸门）
# ---------------------------------------------------------------------------
class AliasEvidenceRefsTest(AliasTestCase):
    def test_default_empty_and_roundtrip(self):
        item = alias_same(evidence_refs=[])
        self.assertEqual(item.evidence_refs, ())
        self.assertEqual(item.to_dict()["evidence_refs"], [])
        self.assertEqual(gir.AliasRelation.from_dict(item.to_dict()).evidence_refs, ())

    def test_order_preserved(self):
        refs = ["fixture:alias/a", "fixture:alias/b", "fixture:alias/c"]
        item = alias_same(evidence_refs=refs)
        self.assertEqual(list(item.evidence_refs), refs)
        self.assertEqual(item.to_dict()["evidence_refs"], refs)
        self.assertEqual(list(gir.AliasRelation.from_dict(item.to_dict()).evidence_refs), refs)
        reversed_item = alias_same(evidence_refs=list(reversed(refs)))
        self.assertNotEqual(canonical(item), canonical(reversed_item))

    def test_items_must_be_nonempty_strings_with_index(self):
        for bad, needle in (([None], "evidence_refs[0]"), ([""], "evidence_refs[0]"),
                            (["   "], "evidence_refs[0]"), ([5], "evidence_refs[0]"),
                            ([[]], "evidence_refs[0]"),
                            (["ok", None], "evidence_refs[1]")):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: alias_same(evidence_refs=b), needle
                )

    def test_container_must_be_list_or_tuple(self):
        for bad in ({"fixture:a"}, {"fixture:a": 1}, (x for x in ["fixture:a"])):
            with self.subTest(bad=type(bad).__name__):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: alias_same(evidence_refs=b), "evidence_refs"
                )
                self.assertIn("list/tuple", str(exc))
                self.assertIn(type(bad).__name__, str(exc))

    def test_string_container_is_not_split_into_characters(self):
        exc = self.assertValueErrorLocated(
            lambda: alias_same(evidence_refs="fixture:alias/ev1"), "evidence_refs"
        )
        self.assertIn("list/tuple", str(exc))
        self.assertIn("str", str(exc))

    def test_dict_container_keys_are_not_taken(self):
        exc = self.assertValueErrorLocated(
            lambda: alias_same(evidence_refs={"fixture:alias/ev1": 1}), "evidence_refs"
        )
        self.assertIn("dict", str(exc))

    def test_list_and_tuple_accepted(self):
        self.assertEqual(alias_same(evidence_refs=["fixture:a"]).evidence_refs,
                         ("fixture:a",))
        self.assertEqual(alias_same(evidence_refs=("fixture:a",)).evidence_refs,
                         ("fixture:a",))

    def test_duplicates_allowed(self):
        item = alias_same(evidence_refs=["fixture:same", "fixture:same"])
        self.assertEqual(item.evidence_refs, ("fixture:same", "fixture:same"))
        self.assertEqual(gir.AliasRelation.from_dict(item.to_dict()).evidence_refs,
                         ("fixture:same", "fixture:same"))

    def test_evidence_refs_frozen_to_tuple_with_defensive_copy(self):
        refs = ["fixture:alias/ev1"]
        item = alias_same(evidence_refs=refs)
        self.assertIsInstance(item.evidence_refs, tuple)
        refs.append("fixture:alias/ev3")
        self.assertEqual(item.evidence_refs, ("fixture:alias/ev1",))


# ---------------------------------------------------------------------------
# 6. 往返、JSON 数组闸门与字节稳定
# ---------------------------------------------------------------------------
class AliasRoundTripTest(AliasTestCase):
    def test_roundtrip_equality_and_dict_equality(self):
        for item in (alias_same(), alias_view(), alias_slice()):
            with self.subTest(kind=item.relation_kind):
                back = gir.AliasRelation.from_dict(item.to_dict())
                self.assertEqual(back, item)
                self.assertEqual(back.to_dict(), item.to_dict())

    def test_repeated_canonical_serialization_byte_identical(self):
        for item in (alias_same(), alias_view()):
            with self.subTest(kind=item.relation_kind):
                first = canonical(item)
                self.assertEqual(first.encode("utf-8"), canonical(item).encode("utf-8"))
                self.assertEqual(first,
                                 canonical(gir.AliasRelation.from_dict(item.to_dict())))

    def test_canonical_bytes_deterministic_across_processes(self):
        item = alias_view(transform_expr="cache[..., -W:, :]  # 中文注释")
        expected = canonical(item)
        literal = repr(json.dumps(item.to_dict(), ensure_ascii=False))
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.graph_ir import AliasRelation\n"
            f"data = json.loads({literal})\n"
            "a = AliasRelation.from_dict(data)\n"
            "sys.stdout.write(json.dumps(a.to_dict(), ensure_ascii=False, sort_keys=True,"
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
        for bad in ([], "alias", 3, None, True):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: gir.AliasRelation.from_dict(b), "别名关系", "对象"
                )

    def test_from_dict_missing_required_fields_named(self):
        for missing in ALIAS_REQUIRED:
            payload = alias_view().to_dict()
            del payload[missing]
            with self.subTest(missing=missing):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.AliasRelation.from_dict(p), missing
                )

    def test_from_dict_evidence_refs_must_be_json_array(self):
        for bad in (None, 5, True, "ev1", "", {"fixture:a": 1}):
            payload = alias_view().to_dict()
            payload["evidence_refs"] = bad
            with self.subTest(bad=bad):
                exc = self.assertValueErrorLocated(
                    lambda p=payload: gir.AliasRelation.from_dict(p), "evidence_refs"
                )
                self.assertIn("JSON 数组", str(exc))

    def test_from_dict_enforces_kind_invariants(self):
        payload = alias_view().to_dict()
        payload["base_tensor_id"] = payload["alias_tensor_id"]
        self.assertValueErrorLocated(lambda: gir.AliasRelation.from_dict(payload),
                                     "base_tensor_id", "alias_tensor_id")
        payload2 = alias_same().to_dict()
        payload2["transform_expr"] = "x"
        self.assertValueErrorLocated(lambda: gir.AliasRelation.from_dict(payload2),
                                     "transform_expr")
        payload3 = alias_same().to_dict()
        del payload3["alias_role"]
        self.assertValueErrorLocated(lambda: gir.AliasRelation.from_dict(payload3),
                                     "alias_role")

    def test_from_dict_optional_keys_absent_default(self):
        payload = {"stable_id": "a0", "base_tensor_id": SHARED_TENSOR_ID,
                   "alias_tensor_id": SHARED_TENSOR_ID, "relation_kind": "same_tensor",
                   "alias_role": "key"}
        item = gir.AliasRelation.from_dict(payload)
        self.assertIsNone(item.transform_expr)
        self.assertEqual(item.evidence_refs, ())

    def test_from_dict_extra_keys_ignored(self):
        payload = alias_view().to_dict()
        payload["shape_compatible"] = True
        payload["source_verified"] = False
        self.assertEqual(gir.AliasRelation.from_dict(payload), alias_view())
        self.assertEqual(gir.AliasRelation.from_dict(payload).to_dict(), alias_view().to_dict())

    def test_distinct_aliases_do_not_collapse(self):
        a = alias_same(stable_id="a_a")
        b = alias_same(stable_id="a_b")
        c = alias_same(stable_id="a_a", alias_role="value")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len({a, b, c}), 3)


# ---------------------------------------------------------------------------
# 7. 图级缺口（本单不通过、不声称）
# ---------------------------------------------------------------------------
class AliasGraphLevelGapsTest(AliasTestCase):
    def test_tensor_reference_existence_not_checked(self):
        item = alias_view(base_tensor_id="tensor:not_in_any_graph",
                          alias_tensor_id="tensor:also_missing")
        self.assertEqual(item.base_tensor_id, "tensor:not_in_any_graph")
        self.assertEqual(gir.AliasRelation.from_dict(item.to_dict()), item)

    def test_duplicate_role_binding_not_checked(self):
        """图级缺口：同一 tensor 上重复绑定同一角色本地不检查。"""
        first = alias_same(stable_id="a1", alias_role="key")
        second = alias_same(stable_id="a2", alias_role="key")
        self.assertEqual((first.base_tensor_id, first.alias_role),
                         (second.base_tensor_id, second.alias_role))
        self.assertNotEqual(first, second)

    def test_conflicting_bindings_not_checked(self):
        """图级缺口：同一 alias tensor 绑到两个不同 base 本地不检查。"""
        a = alias_view(stable_id="a1", base_tensor_id="tensor:base_a",
                       alias_tensor_id="tensor:alias_z")
        b = alias_view(stable_id="a2", base_tensor_id="tensor:base_b",
                       alias_tensor_id="tensor:alias_z")
        self.assertEqual(a.alias_tensor_id, b.alias_tensor_id)
        self.assertNotEqual(a.base_tensor_id, b.base_tensor_id)
        self.assertNotEqual(a, b)

    def test_source_evidence_and_physical_memory_truth_not_checked(self):
        item = alias_same(evidence_refs=["fixture:claimed_shared_storage"])
        self.assertEqual(gir.AliasRelation.from_dict(item.to_dict()), item)
        payload = item.to_dict()
        for forbidden in ("memory_address", "storage_verified", "source_checked",
                          "physical_alias_verified"):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, payload)

    def test_view_slice_axis_transform_compatibility_not_checked(self):
        """图级缺口：view/slice 的轴变换是否与两端张量轴相容，本地无法判断。"""
        item = alias_slice(transform_expr="totally unrelated symbolic text")
        self.assertEqual(item.transform_expr, "totally unrelated symbolic text")
        self.assertEqual(gir.AliasRelation.from_dict(item.to_dict()), item)
        self.assertNotIn("axis_compatible", item.to_dict())

    def test_graph_level_stable_id_uniqueness_not_enforced(self):
        first = alias_same(stable_id="a_dup")
        second = alias_same(stable_id="a_dup")
        self.assertEqual(first, second)
        self.assertEqual(len([first, second]), 2)

    def test_transform_expr_not_evaluated_or_normalized(self):
        """只存原文：不做符号求值/化简，往返后逐字一致（含空格与注释）。"""
        expr = "cache[..., -W:, :]  # keep  exactly"
        item = alias_view(transform_expr=expr)
        self.assertEqual(gir.AliasRelation.from_dict(item.to_dict()).transform_expr, expr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
