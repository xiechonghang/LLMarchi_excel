"""S1 01a Node 探针独立验证：显式 stable_id、端口方向/顺序、可选引用、往返与 frozen。

范围严格限定于 `results/development_01a_node_probe.md` 声明的 Node 增量：
- `stable_id` 是**显式必填字段**（非派生）；`op_type`/`granularity` 非空字符串；
- `input_ports`/`output_ports` 按方向分列、构造时冻结为 tuple、**顺序即端口顺序**，
  每个端口 `direction` 必须与所在列表一致，各列表内端口名不得重复（跨方向同名允许）；
- `evidence_refs` 为可选有序引用，各项非空字符串，顺序原样保留；
- `to_dict`/`from_dict` 稳定往返（dataclass 相等、重复与跨进程字节一致）；frozen。

**不在本测试范围（只有图级/模板级才能查，本探针无上下文）**：
- `stable_id` 的模板局部唯一性（同 op_type 多次调用不得共用 ID）——本文件无模板/图，
  同名 ID 的两个 Node 均可构造，见 `NodeIdentityTest`；
- Port↔Tensor 引用完整性（`Port.tensor_id` 是否指向真实张量）；
- 缺端口/缺节点/缺张量的连接核验、图级 canonical 序列化、guard 三态、state、alias、
  fold、template/instance、JSON Schema 与 01b 检查器、真实 DeepSeek 结构提取。

纪律：
- 数据为**合成契约样例（synthetic contract fixture）**，非 DeepSeek 结构；
- 期望值来自派工文档与开发报告声明的接口，不取自产品生成的输出（不复制产品输出为 golden）；
- 只用标准库；不安装/导入 jsonschema、torch、transformers、safetensors；
- `NodeFromDictBoundaryTest` 中的 7 项按契约要求断言"可分类 + 字段级"错误；产品修复单
  `development_01a_node_json_arrays_fix.md`（SHA `27deafa5…`）落地后**已全部转绿**，
  保留为本修复的回归护栏（发现 F1–F7 见 `verification_01a_node_probe.md` §4）。
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

CANONICAL_KWARGS = {"ensure_ascii": False, "sort_keys": True, "separators": (",", ":")}
NODE_TO_DICT_KEYS = {
    "stable_id", "op_type", "granularity",
    "input_ports", "output_ports", "evidence_refs",
}


def make_port(name, direction, tensor_id=None):
    return gir.Port(name=name, direction=direction,
                    tensor_id=tensor_id if tensor_id is not None else f"tensor:{name}")


def make_node(**overrides):
    """合成契约样例节点：rms_norm 的一次具名调用（非 DeepSeek 结构）。"""
    spec = {
        "stable_id": "n_rms_norm_0",
        "op_type": "rms_norm",
        "granularity": "op",
        "input_ports": [make_port("hidden_states", "input")],
        "output_ports": [make_port("normed", "output")],
        "evidence_refs": ["fixture:synthetic_contract/node/rms_norm_0"],
    }
    spec.update(overrides)
    return gir.Node(**spec)


def make_rich_node(**overrides):
    """多端口双向样例：输入两路、输出两路、两条引用（顺序有意义）。"""
    spec = {
        "stable_id": "n_attention_core_0",
        "op_type": "attention_core",
        "granularity": "op",
        "input_ports": [make_port("q", "input"), make_port("kv", "input")],
        "output_ports": [make_port("attn_out", "output"), make_port("kv_cache", "output")],
        "evidence_refs": ["fixture:synthetic_contract/node/attn/q",
                          "fixture:synthetic_contract/node/attn/kv"],
    }
    spec.update(overrides)
    return gir.Node(**spec)


def canonical(node):
    """仅用于稳定性比较：对 to_dict() 做确定性 JSON 编码（非产品 API）。"""
    return json.dumps(node.to_dict(), **CANONICAL_KWARGS)


class NodeTestCase(unittest.TestCase):
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
# 1. 接口快照：字段、必填/可选、显式 stable_id、frozen
# ---------------------------------------------------------------------------
class NodeInterfaceSnapshotTest(NodeTestCase):
    def test_node_is_frozen_dataclass_with_declared_field_order(self):
        self.assertTrue(gir.Node.__dataclass_params__.frozen)
        self.assertEqual(
            tuple(f.name for f in fields(gir.Node)),
            ("stable_id", "op_type", "granularity",
             "input_ports", "output_ports", "evidence_refs"),
        )

    def test_required_fields_and_optional_defaults(self):
        by_name = {f.name: f for f in fields(gir.Node)}
        for required in ("stable_id", "op_type", "granularity"):
            with self.subTest(required=required):
                self.assertIs(by_name[required].default, MISSING)
        for optional in ("input_ports", "output_ports", "evidence_refs"):
            with self.subTest(optional=optional):
                self.assertEqual(by_name[optional].default, ())
        node = gir.Node(stable_id="n0", op_type="op", granularity="op")
        self.assertEqual(node.input_ports, ())
        self.assertEqual(node.output_ports, ())
        self.assertEqual(node.evidence_refs, ())

    def test_stable_id_is_an_explicit_field_not_a_derived_property(self):
        """与 Tensor.stable_id（派生 property）对照：Node 的 ID 是显式字段。"""
        static = inspect.getattr_static(gir.Node, "stable_id", None)
        self.assertNotIsInstance(static, property)
        self.assertIsInstance(inspect.getattr_static(gir.Tensor, "stable_id"), property)
        node = gir.Node(stable_id="plain_local_id", op_type="op", granularity="op")
        self.assertEqual(node.stable_id, "plain_local_id")  # 不加前缀、不改写

    def test_frozen_prevents_mutation_and_is_hashable(self):
        node = make_node()
        for attribute, new_value in (("stable_id", "other"), ("op_type", "linear"),
                                     ("granularity", "layer"), ("input_ports", ())):
            with self.subTest(attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(node, attribute, new_value)
        self.assertEqual(hash(node), hash(make_node()))
        self.assertEqual(len({node, make_node()}), 1)

    def test_ports_frozen_to_tuple_with_defensive_copy(self):
        inputs = [make_port("hidden_states", "input")]
        outputs = [make_port("normed", "output")]
        refs = ["fixture:synthetic_contract/node/rms_norm_0"]
        node = gir.Node(stable_id="n0", op_type="rms_norm", granularity="op",
                        input_ports=inputs, output_ports=outputs, evidence_refs=refs)
        self.assertIsInstance(node.input_ports, tuple)
        self.assertIsInstance(node.output_ports, tuple)
        self.assertIsInstance(node.evidence_refs, tuple)
        inputs.append(make_port("extra", "input"))
        outputs.append(make_port("extra_out", "output"))
        refs.append("fixture:synthetic_contract/node/extra")
        self.assertEqual(len(node.input_ports), 1)
        self.assertEqual(len(node.output_ports), 1)
        self.assertEqual(len(node.evidence_refs), 1)

    def test_unexpected_constructor_keyword_rejected(self):
        with self.assertRaises(TypeError):
            gir.Node(stable_id="n0", op_type="op", granularity="op", node_id="n0")

    def test_empty_ports_and_refs_roundtrip(self):
        node = gir.Node(stable_id="n0", op_type="op", granularity="op")
        self.assertEqual(gir.Node.from_dict(node.to_dict()), node)
        self.assertEqual(node.to_dict()["input_ports"], [])
        self.assertEqual(node.to_dict()["evidence_refs"], [])


# ---------------------------------------------------------------------------
# 2. 显式 stable_id 的语义与本地局限
# ---------------------------------------------------------------------------
class NodeIdentityTest(NodeTestCase):
    def test_stable_id_preserved_verbatim(self):
        for stable_id in ("n_rms_norm_0", "0", "layer.3.attn.q_proj", "节点-1"):
            with self.subTest(stable_id=stable_id):
                node = make_node(stable_id=stable_id)
                self.assertEqual(node.stable_id, stable_id)
                self.assertEqual(node.to_dict()["stable_id"], stable_id)
                self.assertEqual(gir.Node.from_dict(node.to_dict()).stable_id, stable_id)

    def test_same_op_type_twice_needs_distinct_stable_ids(self):
        first = make_node(stable_id="n_linear_q_0", op_type="linear")
        second = make_node(stable_id="n_linear_kv_0", op_type="linear")
        self.assertEqual(first.op_type, second.op_type)
        self.assertNotEqual(first.stable_id, second.stable_id)
        self.assertNotEqual(first, second)
        self.assertNotEqual(canonical(first), canonical(second))

    def test_stable_id_is_not_derived_from_other_fields(self):
        base = make_node(stable_id="n_a")
        renamed_ports = make_node(stable_id="n_a",
                                  input_ports=[make_port("other_in", "input")],
                                  output_ports=[make_port("other_out", "output")])
        self.assertEqual(base.stable_id, renamed_ports.stable_id)  # ID 与端口无关
        self.assertNotEqual(base, renamed_ports)

    def test_duplicate_stable_ids_are_locally_constructible(self):
        """本地局限（如实记录）：模板/图级唯一性不在此层校验。

        `同算子多次调用不能共用 node ID` 是**调用方义务**，需图级构造或 01b 检查器检出；
        本探针只提供显式 ID 字段，不声称已校验唯一性（开发报告同此声明）。
        """
        first = make_node(stable_id="n_dup")
        second = make_node(stable_id="n_dup")
        self.assertEqual(first, second)
        self.assertEqual(len([first, second]), 2)  # 两个对象都能存在，无本地登记表


# ---------------------------------------------------------------------------
# 3. 必填文本字段校验
# ---------------------------------------------------------------------------
class NodeFieldValidationTest(NodeTestCase):
    def test_required_text_fields_must_be_nonempty_strings(self):
        for field_name in ("stable_id", "op_type", "granularity"):
            for bad in ("", "   ", "\t\n", None, 123, b"x"):
                with self.subTest(field=field_name, bad=bad):
                    self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: make_node(**{f: b}), field_name
                    )

    def test_required_text_fields_reject_surrounding_whitespace(self):
        for field_name in ("stable_id", "op_type", "granularity"):
            for bad in (" value", "value ", "\tvalue", "value\n"):
                with self.subTest(field=field_name, bad=bad):
                    exc = self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: make_node(**{f: b}), field_name
                    )
                    self.assertIn("空白", str(exc))

    def test_granularity_is_free_text_without_vocabulary(self):
        """边界观察：granularity 词表未定义，任意非空字符串被接受（开发报告同此声明）。"""
        for granularity in ("op", "layer", "stage", "whatever-自定义"):
            with self.subTest(granularity=granularity):
                node = make_node(granularity=granularity)
                self.assertEqual(gir.Node.from_dict(node.to_dict()).granularity, granularity)

    def test_op_type_is_free_text_without_vocabulary(self):
        """边界观察：op_type 同样无受控词表（同算子多次调用靠 stable_id 区分）。"""
        node = make_node(op_type="totally.unknown.op")
        self.assertEqual(gir.Node.from_dict(node.to_dict()).op_type, "totally.unknown.op")


# ---------------------------------------------------------------------------
# 4. 端口：方向、顺序、重复名
# ---------------------------------------------------------------------------
class NodePortsTest(NodeTestCase):
    def test_input_and_output_ports_preserved_with_order(self):
        node = make_rich_node()
        payload = node.to_dict()
        self.assertEqual([p["name"] for p in payload["input_ports"]], ["q", "kv"])
        self.assertEqual([p["direction"] for p in payload["input_ports"]], ["input", "input"])
        self.assertEqual([p["name"] for p in payload["output_ports"]], ["attn_out", "kv_cache"])
        self.assertEqual([p["direction"] for p in payload["output_ports"]],
                         ["output", "output"])
        back = gir.Node.from_dict(payload)
        self.assertEqual(back, node)
        self.assertEqual([p.name for p in back.input_ports], ["q", "kv"])
        self.assertEqual([p.name for p in back.output_ports], ["attn_out", "kv_cache"])

    def test_reversed_port_order_is_a_different_fact(self):
        node = make_rich_node()
        swapped = make_rich_node(
            input_ports=[make_port("kv", "input"), make_port("q", "input")],
            output_ports=[make_port("kv_cache", "output"), make_port("attn_out", "output")])
        self.assertNotEqual(node, swapped)
        self.assertNotEqual(canonical(node), canonical(swapped))
        self.assertEqual([p.name for p in gir.Node.from_dict(node.to_dict()).input_ports],
                         ["q", "kv"])
        self.assertEqual([p.name for p in gir.Node.from_dict(swapped.to_dict()).input_ports],
                         ["kv", "q"])

    def test_input_list_rejects_output_direction_port(self):
        exc = self.assertValueErrorLocated(
            lambda: make_node(input_ports=[make_port("wired_wrong", "output")]),
            "input_ports", "wired_wrong", "output",
        )
        self.assertIn("input", str(exc))

    def test_output_list_rejects_input_direction_port(self):
        exc = self.assertValueErrorLocated(
            lambda: make_node(output_ports=[make_port("wired_wrong", "input")]),
            "output_ports", "wired_wrong", "input",
        )
        self.assertIn("output", str(exc))

    def test_direction_mismatch_detected_not_only_for_first_port(self):
        node_ports = [make_port("ok_in", "input"), make_port("bad_out", "output")]
        exc = self.assertValueErrorLocated(lambda: make_node(input_ports=node_ports),
                                           "input_ports[1]", "bad_out")
        self.assertIn("direction", str(exc))

    def test_duplicate_port_name_in_input_list_rejected(self):
        ports = [make_port("hidden_states", "input"),
                 make_port("other", "input"),
                 make_port("hidden_states", "input", tensor_id="tensor:other")]
        exc = self.assertValueErrorLocated(lambda: make_node(input_ports=ports),
                                           "hidden_states")
        self.assertIn("重复", str(exc))

    def test_duplicate_port_name_in_output_list_rejected(self):
        ports = [make_port("out", "output", tensor_id="tensor:a"),
                 make_port("out", "output", tensor_id="tensor:b")]
        self.assertValueErrorLocated(lambda: make_node(output_ports=ports), "out")

    def test_cross_direction_same_port_name_allowed(self):
        node = make_node(input_ports=[make_port("x", "input")],
                         output_ports=[make_port("x", "output")])
        self.assertEqual([p.name for p in node.input_ports], ["x"])
        self.assertEqual([p.name for p in node.output_ports], ["x"])
        self.assertEqual(gir.Node.from_dict(node.to_dict()), node)

    def test_port_lists_must_be_list_or_tuple(self):
        for bad in ({make_port("p", "input")}, "hidden_states",
                    {"hidden_states": "input"}, (p for p in [make_port("p", "input")])):
            with self.subTest(bad=type(bad).__name__):
                self.assertValueErrorLocated(
                    lambda b=bad: make_node(input_ports=b), "input_ports"
                )

    def test_port_elements_must_be_port(self):
        for bad, needle in (([{"name": "p", "direction": "input", "tensor_id": "t"}], "input_ports[0]"),
                            (["p"], "input_ports[0]"),
                            ([None], "input_ports[0]"),
                            ([make_port("ok", "input"), "p"], "input_ports[1]")):
            with self.subTest(needle=needle):
                self.assertValueErrorLocated(lambda b=bad: make_node(input_ports=b), needle)

    def test_ports_not_shared_between_directions(self):
        node = make_rich_node()
        self.assertEqual([p.direction for p in node.input_ports], ["input", "input"])
        self.assertEqual([p.direction for p in node.output_ports], ["output", "output"])
        self.assertFalse(set(node.input_ports) & set(node.output_ports))


# ---------------------------------------------------------------------------
# 5. evidence_refs：可选、有序、非空字符串
# ---------------------------------------------------------------------------
class NodeEvidenceRefsTest(NodeTestCase):
    def test_evidence_refs_default_empty(self):
        node = gir.Node(stable_id="n0", op_type="op", granularity="op")
        self.assertEqual(node.evidence_refs, ())
        self.assertEqual(node.to_dict()["evidence_refs"], [])
        self.assertEqual(gir.Node.from_dict(node.to_dict()).evidence_refs, ())

    def test_evidence_refs_order_preserved(self):
        refs = ["fixture:synthetic_contract/node/a", "fixture:synthetic_contract/node/b"]
        node = make_rich_node(evidence_refs=refs)
        self.assertEqual(list(node.evidence_refs), refs)
        self.assertEqual(node.to_dict()["evidence_refs"], refs)
        self.assertEqual(list(gir.Node.from_dict(node.to_dict()).evidence_refs), refs)
        reversed_node = make_rich_node(evidence_refs=list(reversed(refs)))
        self.assertNotEqual(node, reversed_node)
        self.assertNotEqual(canonical(node), canonical(reversed_node))

    def test_evidence_refs_items_must_be_nonempty_strings_with_index(self):
        """逐项校验：列表内非字符串/空白项被点名拒绝（含索引）。"""
        for bad, needle in (([None], "evidence_refs[0]"), ([""], "evidence_refs[0]"),
                            (["   "], "evidence_refs[0]"), ([5], "evidence_refs[0]"),
                            ([[]], "evidence_refs[0]"),
                            (["ok", None], "evidence_refs[1]")):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: make_node(evidence_refs=b), needle
                )

    def test_evidence_refs_duplicates_allowed(self):
        """边界观察：未规定引用去重，重复项被原样保留。"""
        node = make_node(evidence_refs=["fixture:same", "fixture:same"])
        self.assertEqual(node.evidence_refs, ("fixture:same", "fixture:same"))
        self.assertEqual(gir.Node.from_dict(node.to_dict()).evidence_refs,
                         ("fixture:same", "fixture:same"))

    def test_evidence_refs_only_items_are_type_checked(self):
        """容器闸门：`evidence_refs` 只接受有序 list/tuple；set/dict/生成器被具名拒绝。

        历史（2026-10-12 由 Harness 更新）：本用例原为**故意固定旧宽松行为**的边界
        观察——接受 `set`、`dict`（取值被丢弃、只留键）、生成器并物化为 tuple，
        对应 Harness 发现 B5/F6。产品按修复单加入构造器容器闸门后
        （`ValueError: evidence_refs 必须是有序 list/tuple 引用列表，得到 <类型>`），
        本用例改为**期望拒绝**；方法名保留以便与
        `development_01a_node_json_arrays_fix.md` / `verification_01a_node_probe.md`
        交叉引用。list/tuple 仍合法且顺序保留。
        """
        for bad in ({"fixture:a"}, {"fixture:a": 1}, (ref for ref in ["fixture:a"])):
            with self.subTest(evidence_refs=type(bad).__name__):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: make_node(evidence_refs=b), "evidence_refs"
                )
                self.assertIn("list/tuple", str(exc))
                self.assertIn(type(bad).__name__, str(exc))
        self.assertEqual(make_node(evidence_refs=["fixture:a", "fixture:b"]).evidence_refs,
                         ("fixture:a", "fixture:b"))
        self.assertEqual(make_node(evidence_refs=("fixture:a",)).evidence_refs,
                         ("fixture:a",))


# ---------------------------------------------------------------------------
# 6. to_dict / from_dict 稳定往返与常规负例
# ---------------------------------------------------------------------------
class NodeRoundTripTest(NodeTestCase):
    def test_to_dict_shape_and_keys(self):
        payload = make_rich_node().to_dict()
        self.assertEqual(set(payload), NODE_TO_DICT_KEYS)
        self.assertEqual(set(payload["input_ports"][0]), {"name", "direction", "tensor_id"})
        self.assertEqual(payload["stable_id"], "n_attention_core_0")
        self.assertEqual(payload["op_type"], "attention_core")
        self.assertEqual(payload["granularity"], "op")

    def test_roundtrip_equality_both_directions(self):
        node = make_rich_node()
        back = gir.Node.from_dict(node.to_dict())
        self.assertEqual(back, node)
        self.assertEqual(back.to_dict(), node.to_dict())
        self.assertEqual([p.tensor_id for p in back.input_ports],
                         [p.tensor_id for p in node.input_ports])

    def test_repeated_canonical_serialization_byte_identical(self):
        node = make_rich_node()
        first = canonical(node)
        self.assertEqual(first.encode("utf-8"), canonical(node).encode("utf-8"))
        self.assertEqual(first, canonical(gir.Node.from_dict(node.to_dict())))

    def test_canonical_bytes_deterministic_across_processes(self):
        node = make_rich_node()
        expected = canonical(node)
        literal = repr(json.dumps(node.to_dict(), ensure_ascii=False))
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.graph_ir import Node\n"
            f"data = json.loads({literal})\n"
            "n = Node.from_dict(data)\n"
            "sys.stdout.write(json.dumps(n.to_dict(), ensure_ascii=False, sort_keys=True,"
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
        for bad in ([], "node", 3, None, True):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(lambda b=bad: gir.Node.from_dict(b),
                                             "节点", "对象")

    def test_from_dict_missing_required_fields_named(self):
        for missing in ("stable_id", "op_type", "granularity"):
            payload = make_node().to_dict()
            del payload[missing]
            with self.subTest(missing=missing):
                self.assertValueErrorLocated(lambda p=payload: gir.Node.from_dict(p), missing)

    def test_from_dict_optional_keys_absent_default_to_empty(self):
        payload = {"stable_id": "n0", "op_type": "op", "granularity": "op"}
        node = gir.Node.from_dict(payload)
        self.assertEqual(node.input_ports, ())
        self.assertEqual(node.output_ports, ())
        self.assertEqual(node.evidence_refs, ())

    def test_from_dict_revalidates_nested_ports(self):
        cases = []
        bad_direction = make_node().to_dict()
        bad_direction["input_ports"] = [{"name": "p", "direction": "output", "tensor_id": "t"}]
        cases.append(("方向不符", bad_direction, ("input_ports[0]", "output")))
        missing_field = make_node().to_dict()
        missing_field["output_ports"] = [{"name": "p"}]
        cases.append(("端口缺字段", missing_field, ("端口记录缺少字段",)))
        not_object = make_node().to_dict()
        not_object["input_ports"] = ["p"]
        cases.append(("端口非对象", not_object, ("端口记录", "对象")))
        for label, payload, needles in cases:
            with self.subTest(case=label):
                self.assertValueErrorLocated(lambda p=payload: gir.Node.from_dict(p), *needles)

    def test_from_dict_extra_keys_ignored(self):
        """边界观察：未知多余键被忽略（前向兼容），不影响往返相等。"""
        payload = make_node().to_dict()
        payload["template_id"] = "tpl_rms_norm"
        payload["instance_binding"] = {"layer": 3}
        self.assertEqual(gir.Node.from_dict(payload), make_node())
        self.assertEqual(gir.Node.from_dict(payload).to_dict(), make_node().to_dict())

    def test_distinct_nodes_do_not_collapse(self):
        a = make_node(stable_id="n_a")
        b = make_node(stable_id="n_b")
        c = make_node(stable_id="n_a", op_type="linear")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len({a, b, c}), 3)


# ---------------------------------------------------------------------------
# 7. from_dict 边界（按契约要求断言"可分类 + 字段级"错误）
#
# 本类 7 项对应 Harness 发现 F1–F7（可选列表字段非 JSON 数组时裸 TypeError 逃逸、
# 字符串被逐字符拆解、JSON 对象被取键当引用）。产品修复单
# development_01a_node_json_arrays_fix.md 落地后已全部转绿，本类保留为回归护栏：
# 若 JSON 数组闸门被移除或放宽，这 7 项会重新失败。
# ---------------------------------------------------------------------------
class NodeFromDictBoundaryTest(NodeTestCase):
    def test_from_dict_rejects_null_input_ports(self):
        """F1：`input_ports: null` 应字段级拒绝，而非裸 TypeError。"""
        payload = make_node().to_dict()
        payload["input_ports"] = None
        with self.assertRaises(ValueError) as ctx:
            gir.Node.from_dict(payload)
        self.assertIn("input_ports", str(ctx.exception))

    def test_from_dict_rejects_null_output_ports(self):
        """F2：`output_ports: null` 应字段级拒绝，而非裸 TypeError。"""
        payload = make_node().to_dict()
        payload["output_ports"] = None
        with self.assertRaises(ValueError) as ctx:
            gir.Node.from_dict(payload)
        self.assertIn("output_ports", str(ctx.exception))

    def test_from_dict_rejects_non_sequence_port_lists(self):
        """F3：端口列表为整数等非序列时应字段级拒绝，而非裸 TypeError。"""
        payload = make_node().to_dict()
        payload["input_ports"] = 5
        with self.assertRaises(ValueError) as ctx:
            gir.Node.from_dict(payload)
        self.assertIn("input_ports", str(ctx.exception))

    def test_from_dict_rejects_null_evidence_refs(self):
        """F4：`evidence_refs: null` 应字段级拒绝，而非裸 TypeError。"""
        payload = make_node().to_dict()
        payload["evidence_refs"] = None
        with self.assertRaises(ValueError) as ctx:
            gir.Node.from_dict(payload)
        self.assertIn("evidence_refs", str(ctx.exception))

    def test_from_dict_rejects_non_sequence_evidence_refs(self):
        """F7：`evidence_refs` 为整数时应字段级拒绝，而非裸 TypeError。"""
        payload = make_node().to_dict()
        payload["evidence_refs"] = 5
        with self.assertRaises(ValueError) as ctx:
            gir.Node.from_dict(payload)
        self.assertIn("evidence_refs", str(ctx.exception))

    def test_from_dict_rejects_string_evidence_refs(self):
        """F5：字符串不是引用数组；`"ev1"` 不得被拆成 `('e','v','1')`，`""` 不得当空列表。"""
        for bad in ("ev1", ""):
            payload = make_node().to_dict()
            payload["evidence_refs"] = bad
            with self.subTest(evidence_refs=bad):
                with self.assertRaises(ValueError) as ctx:
                    gir.Node.from_dict(payload)
                self.assertIn("evidence_refs", str(ctx.exception))

    def test_from_dict_rejects_object_evidence_refs(self):
        """F6：JSON 对象不是引用数组，应拒绝，而不是取其键当引用。"""
        payload = make_node().to_dict()
        payload["evidence_refs"] = {"fixture:a": 1, "fixture:b": 2}
        with self.assertRaises(ValueError) as ctx:
            gir.Node.from_dict(payload)
        self.assertIn("evidence_refs", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
