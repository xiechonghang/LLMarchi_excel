"""S1 01a Port 探针独立验证：字段、方向、非法字段拒绝、往返与 frozen。

范围严格限定于 `results/development_01a_port_probe.md` 声明的 Port 增量：
- `Port` frozen dataclass 的字段集与声明顺序（name / direction / tensor_id，全部必填无默认）；
- `direction` 只接受 `input` / `output`（大小写与空白严格），非法值具名拒绝并回显收到的值；
- `name` / `tensor_id` 非空字符串、拒绝首尾空白；
- `to_dict()` / `from_dict()` 稳定往返（dataclass 相等、重复与跨进程字节一致）；
- frozen：赋值抛 `FrozenInstanceError`；可哈希、可作集合/字典键。

**不在本测试范围**（不在此声称已证）：Node / Tensor / Edge / SemanticGraph、图级
canonical 序列化与 `graph_to_json`/`graph_from_json`、Port 稳定 ID 与图级唯一性、
端口到张量的引用解析（本探针无图上下文）、guard 三态、state、alias、fold、
template/instance、JSON Schema 与图检查器、根 API 导出、真实 DeepSeek 结构提取。

纪律：
- 数据为**合成契约样例（synthetic contract fixture）**，非 DeepSeek 结构；
- 期望值来自派工文档与开发报告声明的接口，不取自产品生成的输出（不复制产品输出为 golden）；
- 只用标准库；不安装/导入 jsonschema、torch、transformers、safetensors；
- 不因“尚未实现的类型不存在”而断言（Claude 可能正在并行加入 Tensor 等），
  只固定 Port 自身已声明的契约。
"""

from __future__ import annotations

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

DIRECTIONS = ("input", "output")
CANONICAL_KWARGS = {"ensure_ascii": False, "sort_keys": True, "separators": (",", ":")}


def make_port(**overrides):
    """构造一个合法端口（合成契约样例）。"""
    spec = {
        "name": "hidden_states",
        "direction": "input",
        "tensor_id": "tensor:hidden_states",
    }
    spec.update(overrides)
    return gir.Port(**spec)


def canonical(port):
    """仅用于稳定性比较：对 to_dict() 做确定性 JSON 编码（非产品 API）。"""
    return json.dumps(port.to_dict(), **CANONICAL_KWARGS)


class PortTestCase(unittest.TestCase):
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
# 1. Port 字段与 frozen 接口快照
# ---------------------------------------------------------------------------
class PortInterfaceSnapshotTest(PortTestCase):
    def test_port_is_frozen_dataclass_with_declared_field_order(self):
        self.assertTrue(gir.Port.__dataclass_params__.frozen)
        self.assertEqual(
            tuple(f.name for f in fields(gir.Port)), ("name", "direction", "tensor_id")
        )

    def test_all_fields_required_without_defaults(self):
        for field_info in fields(gir.Port):
            with self.subTest(field=field_info.name):
                self.assertIs(field_info.default, MISSING)
                self.assertIs(field_info.default_factory, MISSING)
        make_port()  # 三字段齐备时合法

    def test_missing_field_raises_naming_the_field(self):
        for missing in ("name", "direction", "tensor_id"):
            kwargs = {"name": "hidden_states", "direction": "input",
                      "tensor_id": "tensor:hidden_states"}
            del kwargs[missing]
            with self.subTest(missing=missing):
                with self.assertRaises(TypeError) as ctx:
                    gir.Port(**kwargs)
                self.assertIn(missing, str(ctx.exception))

    def test_frozen_prevents_mutation(self):
        port = make_port()
        for attribute, new_value in (
            ("name", "other"),
            ("direction", "output"),
            ("tensor_id", "tensor:other"),
        ):
            with self.subTest(attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(port, attribute, new_value)
        self.assertEqual(make_port(), port)  # 未被改动

    def test_equality_and_hashability(self):
        first = make_port()
        second = make_port()
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(len({first, second}), 1)
        self.assertEqual({first: "port"}[second], "port")
        self.assertNotEqual(first, make_port(name="other_states"))

    def test_unexpected_constructor_keyword_rejected(self):
        with self.assertRaises(TypeError):
            gir.Port(name="hidden_states", direction="input",
                     tensor_id="tensor:hidden_states", stable_id="port:hidden_states")


# ---------------------------------------------------------------------------
# 2. input / output 方向
# ---------------------------------------------------------------------------
class PortDirectionTest(PortTestCase):
    def test_input_and_output_accepted_and_preserved(self):
        for direction in DIRECTIONS:
            with self.subTest(direction=direction):
                port = make_port(direction=direction)
                self.assertEqual(port.direction, direction)
                self.assertEqual(port.to_dict()["direction"], direction)
                self.assertEqual(gir.Port.from_dict(port.to_dict()), port)

    def test_direction_is_case_and_whitespace_sensitive(self):
        for bad in ("Input", "INPUT", "Output", "output ", " input", "input\n", "in", "out"):
            with self.subTest(direction=bad):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: make_port(direction=b), "direction"
                )
                self.assertIn(repr(bad), str(exc))  # 回显收到的值

    def test_invalid_direction_values_rejected_with_field_name_and_value(self):
        for bad in ("inout", "", "   ", None, 1, 0, True, False, ["input"], {"direction": "input"}):
            with self.subTest(direction=bad):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: make_port(direction=b), "direction"
                )
                self.assertIn("input", str(exc))
                self.assertIn("output", str(exc))
                self.assertIn(repr(bad), str(exc))

    def test_direction_validation_reached_through_from_dict(self):
        data = make_port().to_dict()
        data["direction"] = "inout"
        self.assertValueErrorLocated(lambda: gir.Port.from_dict(data), "direction")


# ---------------------------------------------------------------------------
# 3. name / tensor_id 字段校验（非法字段拒绝）
# ---------------------------------------------------------------------------
class PortFieldValidationTest(PortTestCase):
    def test_name_must_be_nonempty_string(self):
        for bad in ("", "   ", "\t\n", None, 123, b"hidden", ["hidden_states"]):
            with self.subTest(name=bad):
                self.assertValueErrorLocated(lambda b=bad: make_port(name=b), "name")

    def test_tensor_id_must_be_nonempty_string(self):
        for bad in ("", "   ", "\t\n", None, 123, b"tensor:x", {"id": "tensor:x"}):
            with self.subTest(tensor_id=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: make_port(tensor_id=b), "tensor_id"
                )

    def test_name_rejects_surrounding_whitespace(self):
        for bad in (" hidden_states", "hidden_states ", "\thidden_states", "hidden_states\n"):
            with self.subTest(name=bad):
                exc = self.assertValueErrorLocated(lambda b=bad: make_port(name=b), "name")
                self.assertIn("空白", str(exc))

    def test_tensor_id_rejects_surrounding_whitespace(self):
        for bad in (" tensor:x", "tensor:x ", "\ttensor:x", "tensor:x\n"):
            with self.subTest(tensor_id=bad):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: make_port(tensor_id=b), "tensor_id"
                )
                self.assertIn("空白", str(exc))

    def test_validation_reached_through_from_dict(self):
        for field_name, bad in (("name", ""), ("name", " padded"),
                                ("tensor_id", "   "), ("tensor_id", "tensor:x ")):
            data = make_port().to_dict()
            data[field_name] = bad
            with self.subTest(field=field_name, bad=bad):
                self.assertValueErrorLocated(
                    lambda d=data: gir.Port.from_dict(d), field_name
                )

    def test_name_allows_internal_whitespace(self):
        """如实记录当前宽松度：只拒绝首尾空白，允许内部空格。

        这是边界观察（与具名轴 symbol 同类），若后续收紧字符集须由协调者重新分发测试单。
        """
        port = make_port(name="hidden states")
        self.assertEqual(port.name, "hidden states")
        self.assertEqual(gir.Port.from_dict(port.to_dict()), port)

    def test_tensor_id_is_not_resolved_or_format_checked(self):
        """如实记录当前范围：Port 不校验 tensor_id 指向的张量是否存在、也无前缀约定。

        引用完整性属图级/检查器批次（开发报告已声明 Port 稳定 ID 与图级唯一性留待后续）。
        """
        port = make_port(tensor_id="不存在的张量/without:prefix")
        self.assertEqual(port.tensor_id, "不存在的张量/without:prefix")
        self.assertEqual(gir.Port.from_dict(port.to_dict()), port)


# ---------------------------------------------------------------------------
# 4. to_dict / from_dict 稳定往返
# ---------------------------------------------------------------------------
class PortRoundTripTest(PortTestCase):
    def test_to_dict_shape(self):
        port = make_port()
        self.assertEqual(
            port.to_dict(),
            {"name": "hidden_states", "direction": "input",
             "tensor_id": "tensor:hidden_states"},
        )
        self.assertEqual(set(port.to_dict()), {"name", "direction", "tensor_id"})

    def test_roundtrip_both_directions(self):
        for direction in DIRECTIONS:
            port = make_port(direction=direction, name=f"port_{direction}",
                             tensor_id=f"tensor:{direction}")
            with self.subTest(direction=direction):
                self.assertEqual(gir.Port.from_dict(port.to_dict()), port)
                self.assertEqual(gir.Port.from_dict(port.to_dict()).to_dict(), port.to_dict())

    def test_repeated_serialization_is_byte_identical(self):
        port = make_port()
        first = canonical(port)
        second = canonical(port)
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))
        self.assertEqual(first, canonical(gir.Port.from_dict(port.to_dict())))

    def test_no_time_or_random_component_across_processes(self):
        expected = canonical(make_port())
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.graph_ir import Port\n"
            "p = Port(name='hidden_states', direction='input', tensor_id='tensor:hidden_states')\n"
            "sys.stdout.write(json.dumps(p.to_dict(), ensure_ascii=False, sort_keys=True,"
            " separators=(',', ':')))\n"
        )
        for seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            with self.subTest(pythonhashseed=seed):
                proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                      text=True, env=env, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout, expected)

    def test_unicode_name_and_tensor_id_roundtrip_raw(self):
        port = make_port(name="隐藏状态", tensor_id="张量:隐藏状态")
        text = canonical(port)
        self.assertIn("隐藏状态", text)
        self.assertNotIn("\\u", text)
        self.assertEqual(gir.Port.from_dict(json.loads(text)), port)

    def test_from_dict_requires_json_object(self):
        for bad in ([], "port", 3, None, True):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(lambda b=bad: gir.Port.from_dict(b), "端口", "对象")

    def test_from_dict_missing_fields_are_named(self):
        for missing in ("name", "direction", "tensor_id"):
            data = make_port().to_dict()
            del data[missing]
            with self.subTest(missing=missing):
                self.assertValueErrorLocated(lambda d=data: gir.Port.from_dict(d), missing)

    def test_from_dict_extra_keys_ignored(self):
        """如实记录：未知多余键被忽略（前向兼容），不影响往返相等。"""
        data = make_port().to_dict()
        data["unknown_field"] = "ignored?"
        data["stable_id"] = "port:hidden_states"
        self.assertEqual(gir.Port.from_dict(data), make_port())
        self.assertEqual(gir.Port.from_dict(data).to_dict(), make_port().to_dict())

    def test_from_dict_roundtrip_of_handwritten_synthetic_records(self):
        records = [
            {"name": "hidden_states", "direction": "input", "tensor_id": "tensor:hidden_states"},
            {"name": "q_out", "direction": "output", "tensor_id": "tensor:q"},
            {"name": "kv_out", "direction": "output", "tensor_id": "tensor:kv"},
        ]
        ports = [gir.Port.from_dict(record) for record in records]
        self.assertEqual([p.name for p in ports], ["hidden_states", "q_out", "kv_out"])
        self.assertEqual([p.direction for p in ports], ["input", "output", "output"])
        self.assertEqual([gir.Port.from_dict(p.to_dict()) for p in ports], ports)

    def test_distinct_ports_do_not_collapse(self):
        as_input = make_port(direction="input")
        as_output = make_port(direction="output")
        self.assertNotEqual(as_input, as_output)
        self.assertNotEqual(canonical(as_input), canonical(as_output))
        self.assertNotEqual(make_port(name="q_out"), make_port(name="kv_out"))
        self.assertNotEqual(make_port(tensor_id="tensor:q"), make_port(tensor_id="tensor:kv"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
