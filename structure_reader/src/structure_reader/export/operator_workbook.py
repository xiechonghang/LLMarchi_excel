"""结构阅读数据 → Excel 工作簿导出器（单层/分组共用的唯一导出实现）。

公开接口：

    export_workbook(workbook_spec: dict) -> bytes
    export_operator_workbook(reading_document: dict) -> bytes   # v1 兼容薄适配

职责边界：
- **一处创建 Workbook / 声明 sheet / 写参数引用与公式 / 保存 / 稳定 ZIP**。
  单层与四层组入口都调用 ``export_workbook``；``export_operator_workbook`` 只把
  既有 ``hf_decoder_reading.v1`` 文档转成同一份 spec（薄适配，不再另建工作簿）。
- ``export_workbook`` 消费模型适配层声明的 sheet 与分组：sheet 数量、名称、分组
  归属由输入声明，不在导出器里固定为 Prefill/Decode 两页；DeepSeek 的固定 scope、
  两阶段与场景默认值留在 ``export/hf_workbook.py``。文本模型声明两页；单页能力由
  测试 fixture 验证，不代表 HiDream 已接入。
- 每个 sheet 先写**可填写的性能配置**（seqlen、kvcachelen、batch size、精度、
  存储位宽等示例值，明确可改），再按分组写参数区与算子区。参数/算子引用只在本组
  内解析；派生参数的值写成**真实 Excel 公式**（引用本表输入或本组参数单元格），
  输入未给定或表达式超出白名单时写“未知（原因）”，绝不默认成 0。
- 算子区保留符号 shape、参数联查、真实层组循环归属；性能公式尚未提供时成本列写
  “待提供”，不生成猜测耗时、不把缺失成本记 0、不用忽略空白的 SUM 冒充汇总。
- 输入不合法（缺字段、枚举越界、引用无法解析、id 重复、sheet/分组声明不合法）时
  抛 ``WorkbookExportError``（ValueError 子类），不静默补造缺失事实。
- 相同输入产出相同字节：固定工作簿属性时间与 ZIP 条目顺序/时间戳/权限。

依赖：openpyxl（工程 Python 已可发现）。缺失时直接抛 ImportError，绝不降级为 CSV。
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from datetime import datetime
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.hyperlink import Hyperlink

from structure_reader.scenario_evaluator import (
    ExpressionError,
    check_expression,
    excel_formula,
    expression_names,
)

__all__ = [
    "export_workbook",
    "export_operator_workbook",
    "WorkbookExportError",
    "WorkbookSpecError",
    "ReadingDocumentError",
    "SCHEMA",
    "WORKBOOK_SCHEMA",
]

SCHEMA = "hf_decoder_reading.v1"              # 既有兼容接口（v1）
WORKBOOK_SCHEMA = "operator_workbook.v2"      # 通用工作簿 spec
REQUIRED_PHASE_IDS = ("prefill", "decode")
SHEET_NAMES = {"prefill": "Prefill", "decode": "Decode"}

TENSOR_KINDS = ("activation", "weight", "cache", "mask", "index", "boundary")
SOURCE_KINDS = ("config", "default", "derived", "source", "runtime")
ANALYSIS_STATUSES = ("analyzed", "internal_unanalyzed")
STATUS_CN = {"analyzed": "已分析", "internal_unanalyzed": "内部未分析"}

# 稳定字节：固定属性时间与 ZIP 时间（ZIP 规范纪元 1980，取零时刻）。
_FIXED_EPOCH = datetime(1980, 1, 1, 0, 0, 0)
_FIXED_ZIP_DATE = (1980, 1, 1, 0, 0, 0)
_FIXED_ISO = "1980-01-01T00:00:00Z"
_FIXED_CREATOR = "structure_reader.export.operator_workbook"
# openpyxl 在 save 内部会把 properties.modified 覆写为保存时刻（秒级），
# 因此时间属性必须在稳定化阶段钉回固定值，否则字节随时间漂移、破坏内容寻址。
_CORE_TIME_RE = re.compile(r"(<dcterms:(created|modified)[^>]*>)(.*?)(</dcterms:\2>)", re.S)

_MAX_INLINE_REFS = 12   # 联查公式内最多展开的引用条数，超出只提示剩余项数
_FORMULA_LIMIT = 8000   # Excel 公式上限 8192，留余量后回退为纯文本


# ---------------------------------------------------------------- 异常


class WorkbookExportError(ValueError):
    """导出器错误基类（保持 ValueError 语义，调用方可按 ValueError 捕获）。"""


class ReadingDocumentError(WorkbookExportError):
    """输入阅读数据不符合冻结接口。"""


class WorkbookSpecError(WorkbookExportError):
    """通用工作簿 spec 不合法（sheet/分组/输入/引用声明错误）。"""


# ---------------------------------------------------------------- 校验工具


def _fail(path: str, message: str) -> None:
    raise ReadingDocumentError(f"{path}: {message}")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _plain_str(value, path: str, label: str) -> str:
    if not isinstance(value, str):
        _fail(path, f"{label} 必须是字符串，实得 {type(value).__name__}")
    if ILLEGAL_CHARACTERS_RE.search(value):
        _fail(path, f"{label} 含 XML 非法控制字符，无法写入 xlsx")
    return value


def _require_mapping(value, path: str) -> dict:
    if not isinstance(value, dict):
        _fail(path, f"必须是对象(dict)，实得 {type(value).__name__}")
    return value


def _need(obj: dict, key: str, path: str):
    if key not in obj:
        _fail(path, f"缺少必需字段 {key!r}")
    return obj[key]


def _need_str(obj: dict, key: str, path: str) -> str:
    return _plain_str(_need(obj, key, path), f"{path}.{key}", key)


def _optional_str(obj: dict, key: str, path: str) -> str:
    """可选模型特有字段：缺省写“未声明”，不要求通用输入伪造 HF 字段。"""
    value = obj.get(key, "（未声明）")
    return _plain_str(value, f"{path}.{key}", key)


def _need_list(obj: dict, key: str, path: str) -> list:
    value = _need(obj, key, path)
    if not isinstance(value, list):
        _fail(f"{path}.{key}", f"必须是数组，实得 {type(value).__name__}")
    return value


def _str_list(obj: dict, key: str, path: str) -> list:
    items = _need_list(obj, key, path)
    return [_plain_str(item, f"{path}.{key}[{i}]", key) for i, item in enumerate(items)]


def _check_enum(value: str, allowed: tuple, path: str, label: str) -> str:
    if value not in allowed:
        _fail(path, f"{label} 必须是 {'/'.join(allowed)} 之一，实得 {value!r}")
    return value


def _json_value(value, path: str):
    """参数值允许 JSON 标量/数组/对象/null；其它 Python 类型一律拒绝。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        checked = {}
        for key, item in value.items():
            name = _plain_str(str(key), f"{path}.{key}", "对象键")
            checked[name] = _json_value(item, f"{path}.{name}")
        return checked
    _fail(path, f"参数值必须是 JSON 标量/数组/对象或 null，实得 {type(value).__name__}")


def _canonical(value) -> str:
    """值的确定性字符串化：复合值用合法 JSON（对象键排序），保证字节稳定。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _value_kind(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "布尔"
    if isinstance(value, (int, float)):
        return "数值"
    if isinstance(value, str):
        return "文本"
    if isinstance(value, list):
        return "数组"
    if isinstance(value, dict):
        return "对象"
    return "未知"  # 校验后不可达


# ---------------------------------------------------------------- 文档解析


def _validate_source_refs(raw_list, path: str) -> list:
    refs = []
    for i, raw in enumerate(raw_list):
        item_path = f"{path}[{i}]"
        obj = _require_mapping(raw, item_path)
        refs.append(
            {
                "path": _need_str(obj, "path", item_path),
                "locator": _need_str(obj, "locator", item_path),
                "kind": _check_enum(
                    _need_str(obj, "kind", item_path), SOURCE_KINDS,
                    f"{item_path}.kind", "来源类型"),
                "note": _need_str(obj, "note", item_path),
            }
        )
    return refs


def _dedup(items: list) -> list:
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _validate_parameters(raw_list, path: str) -> list:
    params = []
    seen = {}
    for i, raw in enumerate(raw_list):
        item_path = f"{path}[{i}]"
        obj = _require_mapping(raw, item_path)
        pid = _need_str(obj, "id", item_path)
        if pid in seen:
            _fail(f"{item_path}.id", f"参数 id {pid!r} 在本阶段重复（先见于 {seen[pid]}）")
        seen[pid] = item_path
        expression = _need(obj, "expression", item_path)
        if expression is not None and not isinstance(expression, str):
            _fail(f"{item_path}.expression", "expression 必须是公式字符串或 null")
        value_ref = obj.get("value_ref")
        formula = obj.get("formula")
        unknown_reason = obj.get("unknown_reason", "")
        axis_aliases = obj.get("axis_aliases", [])
        domain = obj.get("domain")
        if domain is not None:
            if not isinstance(domain, dict):
                _fail(f"{item_path}.domain", "domain 必须是对象或 null")
            if domain.get("kind") != "integer":
                _fail(f"{item_path}.domain.kind", "domain.kind 目前只支持 integer")
            minimum = domain.get("min")
            if minimum is not None and (isinstance(minimum, bool) or not isinstance(minimum, int)):
                _fail(f"{item_path}.domain.min", "domain.min 必须是整数")
            max_symbol = domain.get("max_symbol")
            if max_symbol is not None and not isinstance(max_symbol, str):
                _fail(f"{item_path}.domain.max_symbol", "domain.max_symbol 必须是字符串")
        consistency = obj.get("consistency")
        if consistency is not None:
            if not isinstance(consistency, str) or not consistency.strip():
                _fail(f"{item_path}.consistency", "consistency 必须是非空白名单布尔表达式或 null")
            try:
                check_expression(consistency)
            except ExpressionError as exc:
                _fail(f"{item_path}.consistency", f"非法表达式: {exc}")
        if not isinstance(axis_aliases, list):
            _fail(f"{item_path}.axis_aliases", "axis_aliases 必须是字符串数组")
        axis_aliases = [
            _plain_str(alias, f"{item_path}.axis_aliases[{i}]", "轴别名")
            for i, alias in enumerate(axis_aliases)
        ]
        for key, value in (("value_ref", value_ref), ("formula", formula),
                           ("unknown_reason", unknown_reason)):
            if value is not None and not isinstance(value, str):
                _fail(f"{item_path}.{key}", "必须是字符串或 null")
        if value_ref is not None and formula is not None:
            _fail(item_path, "value_ref 与 formula 只能声明一个取值来源")
        params.append(
            {
                "id": pid,
                "symbol": _need_str(obj, "symbol", item_path),
                "meaning": _need_str(obj, "meaning", item_path),
                "value": _json_value(_need(obj, "value", item_path), f"{item_path}.value"),
                "expression": expression,
                "note": _need_str(obj, "note", item_path),
                "source_refs": _validate_source_refs(
                    _need_list(obj, "source_refs", item_path), f"{item_path}.source_refs"),
                "value_ref": value_ref,
                "formula": formula,
                "unknown_reason": unknown_reason or "",
                "axis_aliases": axis_aliases,
                **({"domain": dict(domain)} if domain is not None else {}),
                **({"consistency": consistency} if consistency is not None else {}),
            }
        )
    return params


def _validate_shape(raw_shape, path: str) -> list:
    if not isinstance(raw_shape, list):
        _fail(path, f"shape 必须是按轴顺序的数组，实得 {type(raw_shape).__name__}")
    axes = []
    for i, axis in enumerate(raw_shape):
        axis_path = f"{path}[{i}]"
        if _is_int(axis):
            axes.append(axis)
        elif isinstance(axis, str):
            text = _plain_str(axis, axis_path, "shape 轴")
            if not text.strip():
                _fail(axis_path, "shape 轴表达式不能为空")
            axes.append(text)
        else:
            _fail(axis_path, "shape 轴必须是整数或符号表达式字符串，实得 "
                             f"{type(axis).__name__}")
    return axes


def _validate_tensors(raw_list, path: str) -> list:
    tensors = []
    seen = {}
    for i, raw in enumerate(raw_list):
        item_path = f"{path}[{i}]"
        obj = _require_mapping(raw, item_path)
        tid = _need_str(obj, "id", item_path)
        if tid in seen:
            _fail(f"{item_path}.id", f"张量 id {tid!r} 在本阶段重复（先见于 {seen[tid]}）")
        seen[tid] = item_path
        tensors.append(
            {
                "id": tid,
                "name": _need_str(obj, "name", item_path),
                "shape": _validate_shape(_need(obj, "shape", item_path), f"{item_path}.shape"),
                "parameter_refs": _str_list(obj, "parameter_refs", item_path),
                "kind": _check_enum(
                    _need_str(obj, "kind", item_path), TENSOR_KINDS,
                    f"{item_path}.kind", "张量类型"),
                "note": _need_str(obj, "note", item_path),
            }
        )
    return tensors


def _validate_operators(raw_list, path: str) -> list:
    operators = []
    seen = {}
    for i, raw in enumerate(raw_list):
        item_path = f"{path}[{i}]"
        obj = _require_mapping(raw, item_path)
        oid = _need_str(obj, "id", item_path)
        if oid in seen:
            _fail(f"{item_path}.id", f"算子 id {oid!r} 在本阶段重复（先见于 {seen[oid]}）")
        seen[oid] = item_path
        operators.append(
            {
                "id": oid,
                "module_path": _need_str(obj, "module_path", item_path),
                "name": _need_str(obj, "name", item_path),
                "op": _need_str(obj, "op", item_path),
                "inputs": _str_list(obj, "inputs", item_path),
                "outputs": _str_list(obj, "outputs", item_path),
                "weights": _str_list(obj, "weights", item_path),
                "parameter_refs": _str_list(obj, "parameter_refs", item_path),
                "source_refs": _validate_source_refs(
                    _need_list(obj, "source_refs", item_path), f"{item_path}.source_refs"),
                "analysis_status": _check_enum(
                    _need_str(obj, "analysis_status", item_path), ANALYSIS_STATUSES,
                    f"{item_path}.analysis_status", "分析状态"),
                "note": _need_str(obj, "note", item_path),
                # 原始性能声明，由所在组按是否启用性能区校验（不在此处猜测类别）。
                "performance_raw": obj.get("performance"),
            }
        )
    return operators


def _resolve(refs: list, index: dict, kind: str, path: str) -> None:
    for ref in refs:
        if ref not in index:
            _fail(path, f"{kind}引用 {ref!r} 在本阶段内无法解析（不猜测、不补造）")


def _validate_phase(raw_phase, position: int) -> dict:
    path = f"phases[{position}]"
    obj = _require_mapping(raw_phase, path)
    pid = _need_str(obj, "id", f"{path}.id")
    expected = REQUIRED_PHASE_IDS[position]
    if pid != expected:
        _fail(f"{path}.id", f"第 {position + 1} 个阶段 id 必须是 {expected!r}，实得 {pid!r}")
    title = _need_str(obj, "title", f"{path}.title")
    params = _validate_parameters(_need_list(obj, "parameters", path), f"{path}.parameters")
    tensors = _validate_tensors(_need_list(obj, "tensors", path), f"{path}.tensors")
    operators = _validate_operators(_need_list(obj, "operators", path), f"{path}.operators")

    param_index = {p["id"]: p for p in params}
    tensor_index = {t["id"]: t for t in tensors}

    for tensor in tensors:
        _resolve(tensor["parameter_refs"], param_index, "参数",
                 f"张量 {tensor['id']}.parameter_refs")
    for op in operators:
        _resolve(op["inputs"], tensor_index, "输入张量", f"算子 {op['id']}.inputs")
        _resolve(op["outputs"], tensor_index, "输出张量", f"算子 {op['id']}.outputs")
        _resolve(op["weights"], tensor_index, "权重张量", f"算子 {op['id']}.weights")
        _resolve(op["parameter_refs"], param_index, "参数", f"算子 {op['id']}.parameter_refs")

    return {
        "id": pid,
        "title": title,
        "sheet_name": SHEET_NAMES[pid],
        "parameters": params,
        "tensors": tensors,
        "operators": operators,
        "cache_notes": _str_list(obj, "cache_notes", path),
        "coverage_notes": _str_list(obj, "coverage_notes", path),
        "param_index": param_index,
        "tensor_index": tensor_index,
    }


def _validate_document(reading_document) -> dict:
    if not isinstance(reading_document, dict):
        _fail("reading_document",
              f"必须是对象(dict)，实得 {type(reading_document).__name__}")
    doc = reading_document
    if _need_str(doc, "schema", "schema") != SCHEMA:
        _fail("schema", f"必须是 {SCHEMA!r}")

    model = _require_mapping(_need(doc, "model", "model"), "model")
    model_out = {
        key: _need_str(model, key, "model")
        for key in ("model_key", "display_name", "implementation", "source_revision", "config_path")
    }
    if model_out["implementation"] != "transformers":
        _fail("model.implementation",
              f"接口 v1 固定为 'transformers'，实得 {model_out['implementation']!r}")

    scope = _require_mapping(_need(doc, "scope", "scope"), "scope")
    layer_index = _need(scope, "layer_index", "scope")
    if not _is_int(layer_index) or layer_index < 0:
        _fail("scope.layer_index", "必须是非负整数")
    scope_out = {
        "layer_index": layer_index,
        "attention_type": _need_str(scope, "attention_type", "scope"),
        "moe_type": _need_str(scope, "moe_type", "scope"),
        "analysis_path": _need_str(scope, "analysis_path", "scope"),
        "notes": _str_list(scope, "notes", "scope"),
    }
    if scope_out["analysis_path"] != "eager":
        _fail("scope.analysis_path",
              f"接口 v1 固定为 'eager'，实得 {scope_out['analysis_path']!r}")

    raw_phases = _need_list(doc, "phases", "phases")
    if len(raw_phases) != 2:
        _fail("phases", "必须且只能是 prefill、decode 两个阶段，实得 "
                        f"{len(raw_phases)} 个（禁止第三个阶段或重复阶段）")
    ids = [_require_mapping(p, f"phases[{i}]").get("id") for i, p in enumerate(raw_phases)]
    if len(set(map(str, ids))) != 2:
        _fail("phases", f"阶段 id 重复：{ids!r}")

    return {
        "model": model_out,
        "scope": scope_out,
        "phases": [_validate_phase(p, i) for i, p in enumerate(raw_phases)],
    }


# ---------------------------------------------------------------- 通用 spec 解析


INPUT_KINDS = ("integer", "number", "text", "boolean")


def _validate_inputs(raw_list, path: str) -> list:
    inputs = []
    seen = {}
    for i, raw in enumerate(raw_list):
        item_path = f"{path}[{i}]"
        obj = _require_mapping(raw, item_path)
        iid = _need_str(obj, "id", item_path)
        if iid in seen:
            _fail(f"{item_path}.id", f"输入 id {iid!r} 在本工作表重复（先见于 {seen[iid]}）")
        seen[iid] = item_path
        editable = obj.get("editable", True)
        if not isinstance(editable, bool):
            _fail(f"{item_path}.editable", "editable 必须是布尔值")
        value = _json_value(_need(obj, "value", item_path), f"{item_path}.value")
        kind = _check_enum(
            _need_str(obj, "kind", item_path), INPUT_KINDS,
            f"{item_path}.kind", "输入类型")
        if value is not None:
            if kind == "integer" and not _is_int(value):
                _fail(f"{item_path}.value", "integer 输入的 value 必须是非布尔整数或 null")
            if kind == "number" and (isinstance(value, bool)
                                     or not isinstance(value, (int, float))):
                _fail(f"{item_path}.value", "number 输入的 value 必须是数值或 null")
            if kind == "boolean" and not isinstance(value, bool):
                _fail(f"{item_path}.value", "boolean 输入的 value 必须是布尔值或 null")
            if kind == "text" and not isinstance(value, str):
                _fail(f"{item_path}.value", "text 输入的 value 必须是字符串或 null")
        inputs.append({
            "id": iid,
            "symbol": _need_str(obj, "symbol", item_path),
            "label": _need_str(obj, "label", item_path),
            "value": value,
            "kind": kind,
            "source": _need_str(obj, "source", item_path),
            "note": _need_str(obj, "note", item_path),
            "editable": editable,
        })
    return inputs


def _validate_loop(raw, path: str):
    if raw is None:
        return None
    obj = _require_mapping(raw, path)
    count = _need(obj, "count", path)
    if not _is_int(count) or count < 0:
        _fail(f"{path}.count", "循环次数必须是非负整数")
    return {
        "count": count,
        "label": _need_str(obj, "label", path),
        "condition": _need_str(obj, "condition", path),
        "note": _need_str(obj, "note", path),
    }


def _validate_network(raw, path: str):
    if raw is None:
        return None
    obj = _require_mapping(raw, path)
    periphery = []
    for i, entry in enumerate(_need_list(obj, "periphery", path)):
        item_path = f"{path}.periphery[{i}]"
        item = _require_mapping(entry, item_path)
        periphery.append({
            "id": _need_str(item, "id", item_path),
            "name": _need_str(item, "name", item_path),
            "module_path": _need_str(item, "module_path", item_path),
        })
    groups = []
    for i, entry in enumerate(_need_list(obj, "groups", path)):
        item_path = f"{path}.groups[{i}]"
        item = _require_mapping(entry, item_path)
        layers = item.get("layer_indices", [])
        if not isinstance(layers, list) or not all(
                _is_int(index) and index >= 0 for index in layers):
            _fail(f"{item_path}.layer_indices", "层索引必须是非负整数数组（可省略）")
        count = _need(item, "count", item_path)
        if not _is_int(count) or count < 0:
            _fail(f"{item_path}.count", "层组次数必须是非负整数")
        groups.append({
            "id": _need_str(item, "id", item_path),
            "label": _need_str(item, "label", item_path),
            "layer_indices": layers,
            "count": count,
        })
    layer_count = _need(obj, "layer_count", path)
    if not _is_int(layer_count) or layer_count < 0:
        _fail(f"{path}.layer_count", "整网层数必须是非负整数")
    raw_count_note = obj.get("count_note")
    count_note = (_plain_str(raw_count_note, f"{path}.count_note", "count_note")
                  if raw_count_note is not None else "")
    return {
        "label": _need_str(obj, "label", path),
        "periphery": periphery,
        "groups": groups,
        "layer_count": layer_count,
        "notes": _str_list(obj, "notes", path),
        "cost_status": _need_str(obj, "cost_status", path),
        # 原始串行顺序，由工作表在知道全部组后校验（不在此处按单个组猜）。
        "serial_order": obj.get("serial_order"),
        **({"count_note": count_note} if count_note else {}),
    }


def _perf_expression(value, path: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, f"{label} 必须是非空表达式字符串")
    try:
        check_expression(value)
    except ExpressionError as exc:
        _fail(path, f"{label} 不是白名单表达式: {exc}")
    return value


def _validate_operator_performance(raw, path: str) -> dict:
    """单个算子的 CUBE/VEC 声明（分类、真实维度项、重复次数、条件执行）。"""
    obj = _require_mapping(raw, path)
    kind = _check_enum(_need_str(obj, "class", path), _PERF_CLASSES,
                       f"{path}.class", "性能类别")
    note = _need_str(obj, "note", path)
    unknown_reason = obj.get("unknown_reason", "")
    if unknown_reason is not None and not isinstance(unknown_reason, str):
        _fail(f"{path}.unknown_reason", "必须是字符串或 null")
    if kind == "container":
        return {
            "class": kind,
            "note": note,
            "reason": _need_str(obj, "reason", path),
            "terms": [],
            "repeat": 1,
            "conditional": None,
            "charge": {"mode": "cell"},
            "guards": [],
            "guard_text": "",
            "unknown_reason": unknown_reason or "",
        }
    raw_terms = _need_list(obj, "terms", path)
    if not raw_terms:
        _fail(f"{path}.terms", "至少声明一个计算项（复合算子的每个矩阵项都要列入）")
    terms = []
    for index, raw_term in enumerate(raw_terms):
        term_path = f"{path}.terms[{index}]"
        term = _require_mapping(raw_term, term_path)
        keys = ("b", "m", "k", "n") if kind == "cube" else ("m", "n")
        item = {"label": _need_str(term, "label", term_path)}
        for key in keys:
            item[key] = _perf_expression(
                _need(term, key, term_path), f"{term_path}.{key}", key.upper())
        terms.append(item)
    repeat = obj.get("repeat", 1)
    if isinstance(repeat, bool) or not isinstance(repeat, (int, str)):
        _fail(f"{path}.repeat", "repeat 必须是正整数或白名单表达式字符串")
    if isinstance(repeat, int) and repeat < 1:
        _fail(f"{path}.repeat", "repeat 必须是正整数")
    if isinstance(repeat, str):
        _perf_expression(repeat, f"{path}.repeat", "repeat")
    charge = {"mode": "cell"}
    raw_charge = obj.get("charge")
    if raw_charge is not None:
        charge_obj = _require_mapping(raw_charge, f"{path}.charge")
        mode = _check_enum(_need_str(charge_obj, "mode", f"{path}.charge"),
                           ("excluded",), f"{path}.charge.mode", "计费方式")
        charge = {
            "mode": mode,
            "charged_at": _need_str(charge_obj, "charged_at", f"{path}.charge"),
            "reason": _need_str(charge_obj, "reason", f"{path}.charge"),
        }
    guards = []
    raw_guards = obj.get("guards")
    if raw_guards is not None:
        if not isinstance(raw_guards, list) or not raw_guards:
            _fail(f"{path}.guards", "guards 必须是非空表达式数组")
        for index, guard in enumerate(raw_guards):
            guards.append(_perf_expression(
                guard, f"{path}.guards[{index}]", "guard"))
    guard_text = obj.get("guard_text", "")
    if guard_text is not None and not isinstance(guard_text, str):
        _fail(f"{path}.guard_text", "guard_text 必须是字符串")
    conditional = None
    raw_conditional = obj.get("conditional")
    if raw_conditional is not None:
        conditional_obj = _require_mapping(raw_conditional, f"{path}.conditional")
        conditional = {
            "when": _perf_expression(
                _need(conditional_obj, "when", f"{path}.conditional"),
                f"{path}.conditional.when", "when"),
            "when_text": _need_str(conditional_obj, "when_text", f"{path}.conditional"),
        }
    return {
        "class": kind,
        "note": note,
        "terms": terms,
        "repeat": repeat,
        "conditional": conditional,
        "charge": charge,
        "guards": guards,
        "guard_text": guard_text or "",
        "unknown_reason": unknown_reason or "",
    }


def _validate_segments(raw, path: str, operator_ids: list) -> list:
    """层组内可声明的段（整网外围的入口/循环/出口），必须是算子的顺序划分。"""
    if raw is None:
        return []
    if not isinstance(raw, list):
        _fail(path, f"必须是数组，实得 {type(raw).__name__}")
    segments = []
    seen = set()
    covered: list = []
    for index, raw_segment in enumerate(raw):
        item_path = f"{path}[{index}]"
        item = _require_mapping(raw_segment, item_path)
        sid = _need_str(item, "id", item_path)
        if sid in seen:
            _fail(f"{item_path}.id", f"段 id {sid!r} 在本组重复")
        seen.add(sid)
        subtotal = item.get("subtotal", True)
        if not isinstance(subtotal, bool):
            _fail(f"{item_path}.subtotal", "subtotal 必须是布尔值")
        ids = _str_list(item, "operator_ids", item_path)
        if not ids:
            _fail(f"{item_path}.operator_ids", "段必须至少包含一个算子")
        for oid in ids:
            if oid not in operator_ids:
                _fail(f"{item_path}.operator_ids", f"算子 {oid!r} 不属于本组")
            if oid in covered:
                _fail(f"{item_path}.operator_ids", f"算子 {oid!r} 被多个段重复包含")
        covered.extend(ids)
        segments.append({
            "id": sid,
            "label": _need_str(item, "label", item_path),
            "operator_ids": ids,
            "subtotal": subtotal,
        })
    if covered != list(operator_ids):
        _fail(path, "段必须按算子顺序恰好划分本组算子（不重不漏）")
    return segments


def _validate_engine(raw) -> dict | None:
    if raw is None:
        return None
    obj = _require_mapping(raw, "performance.engine")
    _check_enum(_need_str(obj, "kind", "performance.engine"), (PERF_ENGINE_KIND,),
                "performance.engine.kind", "性能引擎类型")
    cube_id = _need_str(obj, "cube_throughput_input", "performance.engine")
    vec_id = _need_str(obj, "vec_throughput_input", "performance.engine")
    if cube_id == vec_id:
        _fail("performance.engine", "CUBE/VEC 吞吐必须引用两个不同的本表输入")
    defaults = {}
    for key in ("cube_default", "vec_default"):
        value = obj.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            _fail(f"performance.engine.{key}", "必须是正数（默认吞吐 > 0）")
        defaults[key] = value
    note = obj.get("source", "")
    if not isinstance(note, str):
        _fail("performance.engine.source", "必须是字符串")
    return {
        "kind": PERF_ENGINE_KIND,
        "cube_throughput_input": cube_id,
        "vec_throughput_input": vec_id,
        "cube_default": defaults["cube_default"],
        "vec_default": defaults["vec_default"],
        "cube_unit": obj.get("cube_unit", "MAC/ms") if isinstance(obj.get("cube_unit", "MAC/ms"), str) else "MAC/ms",
        "vec_unit": obj.get("vec_unit", "元素/ms") if isinstance(obj.get("vec_unit", "元素/ms"), str) else "元素/ms",
        "source": note,
    }


def _validate_serial_order(raw, path: str, sheet: dict, engine: dict | None) -> list:
    """整网串行求和顺序：恰好覆盖全部层组与声明的段，不重不漏。"""
    if raw is None:
        return []
    if engine is None:
        _fail(path, "未声明 performance.engine 时不应声明串行求和顺序")
    if not isinstance(raw, list):
        _fail(path, f"必须是数组，实得 {type(raw).__name__}")
    groups = {group["id"]: group for group in sheet["groups"]}
    entries = []
    seen_groups = set()
    seen_segments = set()
    for index, raw_entry in enumerate(raw):
        item_path = f"{path}[{index}]"
        item = _require_mapping(raw_entry, item_path)
        kind = _check_enum(_need_str(item, "kind", item_path), ("group", "segment"),
                           f"{item_path}.kind", "串行段类型")
        gid = _need_str(item, "group", item_path)
        condition = None
        raw_condition = item.get("condition")
        if raw_condition is not None:
            condition_obj = _require_mapping(raw_condition, f"{item_path}.condition")
            condition = {
                "when": _perf_expression(
                    _need(condition_obj, "when", f"{item_path}.condition"),
                    f"{item_path}.condition.when", "when"),
                "when_text": _need_str(condition_obj, "when_text",
                                       f"{item_path}.condition"),
                "guards": [_perf_expression(
                    guard, f"{item_path}.condition.guards[{index}]", "guard")
                    for index, guard in enumerate(
                        _str_list(condition_obj, "guards", f"{item_path}.condition")
                        if condition_obj.get("guards") is not None else [])],
            }
        group = groups.get(gid)
        if group is None:
            _fail(f"{item_path}.group", f"引用的组 {gid!r} 不在本工作表")
        if kind == "group":
            if gid in seen_groups:
                _fail(f"{item_path}.group", f"层组 {gid!r} 在串行顺序中重复")
            if group["loop"] is None:
                _fail(f"{item_path}.group", f"组 {gid!r} 未声明循环次数，不能作为层组贡献")
            seen_groups.add(gid)
            entries.append({"kind": "group", "group": gid,
                            **({"condition": condition} if condition else {})})
        else:
            sid = _need_str(item, "segment", item_path)
            segments = {segment["id"]: segment for segment in group.get("segments", [])}
            segment = segments.get(sid)
            if segment is None:
                _fail(f"{item_path}.segment", f"组 {gid!r} 没有段 {sid!r}")
            key = (gid, sid)
            if key in seen_segments:
                _fail(f"{item_path}.segment", f"段 {gid}/{sid} 在串行顺序中重复")
            if not segment["subtotal"]:
                _fail(f"{item_path}.segment",
                      f"段 {gid}/{sid} 声明为不单独计成本，不能作为串行贡献项")
            seen_segments.add(key)
            entries.append({"kind": "segment", "group": gid, "segment": sid,
                            **({"condition": condition} if condition else {})})
    for gid, group in groups.items():
        subtotal_segments = [segment for segment in group.get("segments", [])
                             if segment["subtotal"]]
        # 一个组的成本可以“整组计入”（group 条目）或“按计费段分别计入”
        # （全部 subtotal 段都已列出）；两种方式不能同时用，否则重复计。
        by_segments = bool(subtotal_segments) and all(
            (gid, segment["id"]) in seen_segments for segment in subtotal_segments)
        if group["loop"] is not None and gid not in seen_groups and not by_segments:
            _fail(path, f"层组 {gid!r} 有循环次数却未列入串行求和顺序（不得漏项）")
        if gid in seen_groups and any(
                (gid, segment["id"]) in seen_segments for segment in subtotal_segments):
            _fail(path, f"层组 {gid!r} 不能同时整组与按段计入串行（会重复）")
        for segment in group.get("segments", []):
            if (segment["subtotal"]
                    and (gid, segment["id"]) not in seen_segments
                    and gid not in seen_groups):
                _fail(path, f"段 {gid}/{segment['id']} 声明计成本却未列入串行求和顺序")
    return entries


def _validate_engine_inputs(sheet: dict, engine: dict, path: str) -> None:
    """每页都要有可改的 CUBE/VEC 吞吐输入（缺一即失败，不静默用常量）。"""
    by_id = {item["id"]: item for item in sheet["inputs"]}
    for role, input_id in (("CUBE", engine["cube_throughput_input"]),
                           ("VEC", engine["vec_throughput_input"])):
        item = by_id.get(input_id)
        if item is None:
            _fail(path, f"缺少 {role} 吞吐输入 {input_id!r}（公式必须引用本表可改单元格）")
        if item["kind"] != "number":
            _fail(f"{path}.inputs[{input_id}]", f"{role} 吞吐必须是数值输入")
        if item["value"] is None or isinstance(item["value"], bool) or item["value"] <= 0:
            _fail(f"{path}.inputs[{input_id}]", f"{role} 吞吐默认值必须是正数")
        if not item["editable"]:
            _fail(f"{path}.inputs[{input_id}]", f"{role} 吞吐必须可改")


def _validate_group(raw, position: int, sheet_path: str, input_ids: set,
                    engine: dict | None) -> dict:
    path = f"{sheet_path}.groups[{position}]"
    obj = _require_mapping(raw, path)
    gid = _need_str(obj, "id", path)
    # 通用单页不要求 Decoder 层号：layer_indices/代表层是可选元数据。
    # DeepSeek 适配仍必须提供真实层归属（由 hf_workbook 严格校验）。
    layers = obj.get("layer_indices", [])
    if not isinstance(layers, list):
        _fail(f"{path}.layer_indices", "层索引必须是非负整数数组（可省略）")
    if not all(_is_int(index) and index >= 0 for index in layers):
        _fail(f"{path}.layer_indices", "层索引必须是非负整数数组")
    representative = obj.get("representative_layer_index")
    if representative is not None:
        if not _is_int(representative):
            _fail(f"{path}.representative_layer_index", "代表层必须是整数")
        if layers and representative not in layers:
            _fail(f"{path}.representative_layer_index", "代表层必须是本组真实层索引之一")
        if not layers:
            _fail(f"{path}.representative_layer_index",
                  "未声明真实层索引时不应声明代表层（通用单页不伪造 Decoder 层号）")

    params = _validate_parameters(_need_list(obj, "parameters", path), f"{path}.parameters")
    tensors = _validate_tensors(_need_list(obj, "tensors", path), f"{path}.tensors")
    operators = _validate_operators(_need_list(obj, "operators", path), f"{path}.operators")

    param_index = {p["id"]: p for p in params}
    tensor_index = {t["id"]: t for t in tensors}
    for param in params:
        if param["value_ref"] is not None and param["value_ref"] not in input_ids:
            _fail(f"{path}.parameters[{param['id']}].value_ref",
                  f"引用的本表输入 {param['value_ref']!r} 不存在（不跨组、不跨表借用）")
    for tensor in tensors:
        _resolve(tensor["parameter_refs"], param_index, "参数",
                 f"张量 {tensor['id']}.parameter_refs")
    for op in operators:
        _resolve(op["inputs"], tensor_index, "输入张量", f"算子 {op['id']}.inputs")
        _resolve(op["outputs"], tensor_index, "输出张量", f"算子 {op['id']}.outputs")
        _resolve(op["weights"], tensor_index, "权重张量", f"算子 {op['id']}.weights")
        _resolve(op["parameter_refs"], param_index, "参数", f"算子 {op['id']}.parameter_refs")

    # 性能声明：启用性能区时**每个**算子都必须声明 CUBE/VEC 与真实维度；
    # 未启用时不得夹带半套性能列（否则会与“待提供”的既有契约混淆）。
    operator_ids = [op["id"] for op in operators]
    for op in operators:
        raw_perf = op.pop("performance_raw")
        if engine is None:
            if raw_perf is not None:
                _fail(f"{path}.operators[{op['id']}]",
                      "未声明 performance.engine 时不应给出算子性能声明")
            op["performance"] = None
            continue
        if raw_perf is None:
            _fail(f"{path}.operators[{op['id']}]",
                  "启用性能区时每个算子都必须声明 CUBE/VEC 类别与维度（不得漏项）")
        op["performance"] = _validate_operator_performance(
            raw_perf, f"{path}.operators[{op['id']}].performance")

    execution_condition = None
    raw_execution = obj.get("execution_condition")
    if raw_execution is not None:
        if engine is None:
            _fail(f"{path}.execution_condition", "执行条件只在启用性能区时使用")
        execution_obj = _require_mapping(raw_execution, f"{path}.execution_condition")
        execution_condition = {
            "when": _perf_expression(
                _need(execution_obj, "when", f"{path}.execution_condition"),
                f"{path}.execution_condition.when", "when"),
            "when_text": _need_str(execution_obj, "when_text",
                                   f"{path}.execution_condition"),
            "guards": [_perf_expression(
                guard, f"{path}.execution_condition.guards[{index}]", "guard")
                for index, guard in enumerate(
                    _str_list(execution_obj, "guards", f"{path}.execution_condition")
                    if execution_obj.get("guards") is not None else [])],
        }

    raw_segments = obj.get("segments")
    if raw_segments is not None and engine is None:
        _fail(f"{path}.segments", "段划分只在启用性能区时使用")
    segments = _validate_segments(raw_segments, f"{path}.segments", operator_ids) \
        if raw_segments is not None else []

    return {
        "id": gid,
        "label": _need_str(obj, "label", path),
        "attention_type": _optional_str(obj, "attention_type", path),
        "moe_type": _optional_str(obj, "moe_type", path),
        "layer_indices": layers,
        "representative_layer_index": representative,
        "loop": _validate_loop(obj.get("loop"), f"{path}.loop"),
        "template_note": _need_str(obj, "template_note", path),
        "scope_note": _need_str(obj, "scope_note", path),
        "parameters": params,
        "tensors": tensors,
        "operators": operators,
        "segments": segments,
        "execution_condition": execution_condition,
        "kind": obj.get("kind", "") if isinstance(obj.get("kind", ""), str) else "",
        "cache_notes": _str_list(obj, "cache_notes", path),
        "coverage_notes": _str_list(obj, "coverage_notes", path),
        "param_index": param_index,
        "tensor_index": tensor_index,
    }


def _validate_sheet(raw, position: int, engine: dict | None) -> dict:
    path = f"sheets[{position}]"
    obj = _require_mapping(raw, path)
    sid = _need_str(obj, "id", path)
    inputs = _validate_inputs(obj.get("inputs", []), f"{path}.inputs")
    input_ids = {item["id"] for item in inputs}
    raw_groups = _need_list(obj, "groups", path)
    if not raw_groups:
        _fail(f"{path}.groups", "每个工作表至少声明一个分组")
    groups = []
    seen = set()
    for i, group in enumerate(raw_groups):
        validated = _validate_group(group, i, path, input_ids, engine)
        if validated["id"] in seen:
            _fail(f"{path}.groups[{i}].id", f"分组 id {validated['id']!r} 在本工作表重复")
        seen.add(validated["id"])
        groups.append(validated)
    name = _need_str(obj, "name", path)
    if len(name) > 31:
        _fail(f"{path}.name", "工作表名不能超过 31 个字符")
    if re.search(r"[\[\]:*?/\\]", name):
        _fail(f"{path}.name", "工作表名包含 Excel 禁用字符")
    network = _validate_network(obj.get("network"), f"{path}.network")
    sheet = {
        "id": sid,
        "name": name,
        "title": _need_str(obj, "title", path),
        "notes": _str_list(obj, "notes", path),
        "inputs": inputs,
        "groups": groups,
        "network": network,
    }
    if network is not None:
        network["serial_order"] = _validate_serial_order(
            network.get("serial_order"), f"{path}.network.serial_order", sheet, engine)
    return sheet


def _validate_workbook_spec(spec) -> dict:
    if not isinstance(spec, dict):
        _fail("workbook_spec", f"必须是对象(dict)，实得 {type(spec).__name__}")
    if _need_str(spec, "schema", "schema") != WORKBOOK_SCHEMA:
        _fail("schema", f"必须是 {WORKBOOK_SCHEMA!r}")

    model = _require_mapping(_need(spec, "model", "model"), "model")
    model_out = {
        key: _need_str(model, key, "model")
        for key in ("model_key", "display_name", "implementation",
                    "source_revision", "config_path")
    }
    performance = _require_mapping(spec.get("performance", {}), "performance")
    raw_status = performance.get("status", "待提供")
    if not isinstance(raw_status, str):
        _fail("performance.status", "必须是字符串")
    raw_note = performance.get("note", "")
    if not isinstance(raw_note, str):
        _fail("performance.note", "必须是字符串")
    engine = _validate_engine(performance.get("engine"))
    disclaimer = spec.get("disclaimer", DISCLAIMER)
    if not isinstance(disclaimer, str):
        _fail("disclaimer", "必须是字符串")

    raw_sheets = _need_list(spec, "sheets", "sheets")
    if not raw_sheets:
        _fail("sheets", "至少声明一个工作表")
    sheets = []
    names, ids = set(), set()
    for i, raw in enumerate(raw_sheets):
        sheet = _validate_sheet(raw, i, engine)
        if sheet["name"] in names:
            _fail(f"sheets[{i}].name", f"工作表名 {sheet['name']!r} 重复")
        if sheet["id"] in ids:
            _fail(f"sheets[{i}].id", f"工作表 id {sheet['id']!r} 重复")
        names.add(sheet["name"])
        ids.add(sheet["id"])
        if engine is not None:
            _validate_engine_inputs(sheet, engine, f"sheets[{i}]")
        sheets.append(sheet)

    return {
        "model": model_out,
        "performance": {"status": raw_status, "note": raw_note, "engine": engine},
        "disclaimer": disclaimer,
        "sheets": sheets,
    }


# ---------------------------------------------------------------- 样式与列定义

_THIN = Side(style="thin", color="FFB3BFC7")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_WRAP_TOP = Alignment(wrap_text=True, vertical="top")
_TOP = Alignment(vertical="top")
_CENTER = Alignment(vertical="center")

_FONT_TITLE = Font(bold=True, size=13, color="FFFFFFFF")
_FONT_SECTION = Font(bold=True, size=11, color="FFFFFFFF")
_FONT_HEAD = Font(bold=True, size=10, color="FF1F2933")
_FONT_LABEL = Font(bold=True, size=10, color="FF1F2933")
_FONT_BODY = Font(size=10, color="FF1F2933")
_FONT_BLACK = Font(bold=True, size=10, color="FF9C0006")
_FONT_LINK = Font(size=10, color="FF0563C1", underline="single")

_FILL_TITLE = PatternFill("solid", fgColor="FF1F4E79")
_FILL_PARAM_SECTION = PatternFill("solid", fgColor="FF215C36")
_FILL_OP_SECTION = PatternFill("solid", fgColor="FF7A3B0E")
_FILL_PARAM_HEAD = PatternFill("solid", fgColor="FFDDEBE0")
_FILL_OP_HEAD = PatternFill("solid", fgColor="FFF2E2D0")
_FILL_LABEL = PatternFill("solid", fgColor="FFEDF1F5")
_FILL_BLACK_CELL = PatternFill("solid", fgColor="FFFFC7CE")
_FILL_BLACK_ROW = PatternFill("solid", fgColor="FFFFF3F3")
_FILL_OK = PatternFill("solid", fgColor="FFE7F1E4")
_FILL_INPUT = PatternFill("solid", fgColor="FFFFF2CC")
_FILL_FORMULA = PatternFill("solid", fgColor="FFDDEBF7")
_FILL_UNKNOWN = PatternFill("solid", fgColor="FFFDE9D9")

PARAM_COLUMNS = (
    ("参数id", 18), ("符号", 14), ("含义", 30), ("值", 20), ("值类型", 10),
    ("表达式", 30), ("来源类型", 12), ("来源定位", 30), ("说明", 42),
)
INPUT_COLUMNS = (
    ("配置id", 18), ("符号", 14), ("含义", 30), ("值（可填写）", 20), ("值类型", 10),
    ("默认依据/表达式", 30), ("可编辑", 12), ("来源", 30), ("说明", 42),
)
OP_COLUMNS = (
    ("黑盒标记", 16), ("算子id", 18), ("模块路径", 30), ("名称", 22), ("算子类型", 16),
    ("输入 名称[shape]", 34), ("输出 名称[shape]", 30), ("权重 名称[shape]", 30),
    ("维度符号→本表配置值", 28), ("参数引用 本表id(可点击定位)", 24),
    ("参数联查 符号=值(公式指向本表参数区)", 34), ("源码/配置位置", 32),
    ("分析状态", 12), ("说明", 44),
    ("循环归属（层组/真实层/次数）", 36), ("成本状态（用户公式待提供）", 36),
    ("实际输入 shape（按本表配置重算）", 40),
    ("实际输出 shape（按本表配置重算）", 36),
    ("实际权重 shape（按本表配置重算）", 36),
)
N_COLS = len(OP_COLUMNS)
COLUMN_WIDTHS = tuple(
    max(
        PARAM_COLUMNS[i][1] if i < len(PARAM_COLUMNS) else 0,
        INPUT_COLUMNS[i][1] if i < len(INPUT_COLUMNS) else 0,
        OP_COLUMNS[i][1],
    )
    for i in range(N_COLS)
)

# ---------------------------------------------------------------- 性能区（可选）
#
# 只有 spec 声明 ``performance.engine`` 时才追加这些列并写用户规则公式；未声明时
# 输出与既有工作簿逐字节一致（GLM/Qwen/HiDream 与旧候选不受影响）。
PERF_ENGINE_KIND = "cube_vec"
PERF_COLUMNS = (
    ("分类（用户规则 CUBE/VEC）", 14),
    ("CUBE B", 14),
    ("M（CUBE 行 / VEC 前导元素）", 16),
    ("CUBE K", 14),
    ("N（CUBE 列 / VEC 末维）", 16),
    ("工作量（CUBE MAC / VEC 元素）", 24),
    ("成本状态（已计/未知/条件0）", 22),
    ("性能假设与维度映射（规则来源）", 62),
)
PERF_FIRST_COL = len(OP_COLUMNS) + 1          # 20 → 分类列
PERF_COST_COL = 16                            # 既有「成本状态」列改为用户规则耗时
_COST_HEADER_ENGINE = ("用户规则耗时 ms（默认 4096/64 吞吐；实际按表内 CUBE/VEC 吞吐：CUBE B·M·K·N/吞吐、VEC M·N/吞吐）")
_NUMBER_FORMAT_COST = "#,##0.000000"   # 需显示 1/4096 ms 这类非零小值
_NUMBER_FORMAT_WORK = "#,##0"
_PERF_CLASSES = ("cube", "vec", "container")
_PERF_CLASS_CN = {"cube": "CUBE", "vec": "VEC", "container": "容器（不计成本）"}


def _layout(engine: dict | None) -> dict:
    """按是否启用性能区给出列定义/宽度/列数；未启用时与既有常量完全一致。"""
    columns = tuple(
        ((_COST_HEADER_ENGINE if index == PERF_COST_COL - 1 else label), width)
        for index, (label, width) in enumerate(OP_COLUMNS)
    ) if engine is not None else OP_COLUMNS
    if engine is not None:
        columns = columns + PERF_COLUMNS
    n_cols = len(columns)
    widths = tuple(
        max(
            PARAM_COLUMNS[i][1] if i < len(PARAM_COLUMNS) else 0,
            INPUT_COLUMNS[i][1] if i < len(INPUT_COLUMNS) else 0,
            columns[i][1],
        )
        for i in range(n_cols)
    )
    return {"columns": columns, "n_cols": n_cols, "widths": widths}

DISCLAIMER = (
    "本工作簿由 operator_workbook.v2 通用导出器生成：结构、参数、层组归属均来自输入的"
    "事实文档；导出器不推断模型语义、不填运行期假设值。可在黄色输入区填写性能配置，"
    "派生值由 Excel 公式重算；未给定或超出白名单的表达式写“未知（原因）”，不默认 0。"
)


# ---------------------------------------------------------------- 单元格写入


def _text(ws, row: int, col: int, value, *, font=None, alignment=None, fill=None, border=True):
    """写文本单元格；把 openpyxl 会误判成公式/错误值的字符串锁回文本类型。"""
    cell = ws.cell(row=row, column=col)
    cell.value = "" if value is None else value
    if isinstance(value, str) or value is None:
        cell.data_type = "s"
    cell.font = font or _FONT_BODY
    cell.alignment = alignment or _WRAP_TOP
    if fill is not None:
        cell.fill = fill
    if border:
        cell.border = _BORDER
    return cell


def _number(ws, row: int, col: int, value: int | float):
    cell = ws.cell(row=row, column=col)
    cell.value = value
    cell.font = _FONT_BODY
    cell.alignment = _TOP
    cell.border = _BORDER
    return cell


def _formula(ws, row: int, col: int, formula: str, *, fill=None):
    cell = ws.cell(row=row, column=col)
    cell.value = formula          # 以 = 开头 → openpyxl 判定为公式（公式栏可见引用）
    cell.font = _FONT_BODY
    cell.alignment = _WRAP_TOP
    if fill is not None:
        cell.fill = fill
    cell.border = _BORDER
    return cell


def _banner(ws, row: int, text: str, fill, big: bool = False,
            n_cols: int = N_COLS) -> None:
    for col in range(1, n_cols + 1):
        cell = ws.cell(row=row, column=col)
        cell.fill = fill
        cell.font = _FONT_TITLE if big else _FONT_SECTION
        cell.alignment = _CENTER
    ws.cell(row=row, column=1).value = text
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=n_cols)
    ws.row_dimensions[row].height = 20 if big else 18


def _header_row(ws, row: int, columns, fill) -> None:
    for idx, (label, _w) in enumerate(columns, start=1):
        cell = ws.cell(row=row, column=idx)
        cell.value = label
        cell.font = _FONT_HEAD
        cell.alignment = _WRAP_TOP
        cell.fill = fill
        cell.border = _BORDER
    ws.row_dimensions[row].height = 32


# ---------------------------------------------------------------- 公式辅助


def _symbol_value_pieces(rows: list, ids: list | None = None) -> list:
    """把本表参数行的符号单元格与值单元格串成 $B$r&"="&$D$r。

    ids 与 rows 同序；条目非空时额外拼上该行的参数id单元格 $A$r（同名符号歧义时
    只用id能分辨哪行是哪个参数），不复制数值。
    """
    pieces = []
    for index, r in enumerate(rows):
        pid = None if ids is None else ids[index]
        if pid:
            pieces.append(f'$A${r}&":"&$B${r}&"="&$D${r}')
        else:
            pieces.append(f'$B${r}&"="&$D${r}')
    return pieces


def _xl_quote(text: str) -> str:
    return '"' + str(text).replace('"', '""') + '"'


def _lookup_formula(rows: list, ids: list | None = None,
                    suffix_text: str | None = None) -> str | None:
    """符号联查公式：值取自本表参数行单元格；suffix_text 作为字面量附注。

    无引用或超出 Excel 公式长度上限时返回 None（由调用方回退为纯文本）。
    """
    if not rows:
        return None
    shown = rows[:_MAX_INLINE_REFS]
    shown_ids = ids[:_MAX_INLINE_REFS] if ids is not None else None
    formula = "=" + ' & "; " & '.join(_symbol_value_pieces(shown, shown_ids))
    if len(rows) > len(shown):
        formula += f' & "; " & {_xl_quote(f"…另有{len(rows) - len(shown)}项，完整清单见「参数引用」列")}'
    if suffix_text:
        formula += f' & "; " & {_xl_quote(suffix_text)}'
    if len(formula) > _FORMULA_LIMIT:
        return None
    return formula


def _tensor_label(tensor: dict) -> str:
    axes = ", ".join(str(axis) for axis in tensor["shape"])
    return f"{tensor['name']}[{axes}]"


def _join_sources(refs: list) -> str:
    if not refs:
        return "（无来源）"
    parts = []
    for ref in refs:
        where = f"{ref['path']}#{ref['locator']}" if ref["path"] and ref["locator"] else (
            ref["path"] or ref["locator"] or "（未提供定位）")
        piece = f"[{ref['kind']}] {where}"
        if ref["note"]:
            piece += f"（{ref['note']}）"
        parts.append(piece)
    return " | ".join(parts)


# ---------------------------------------------------------------- 性能区公式


def _perf_symbol_table(group: dict, param_cells: dict, param_guards: dict,
                       input_rows: dict, inputs: dict) -> dict:
    """性能表达式可引用的符号/参数 id → (单元格, 活校验)。

    本组参数优先（含 id 与轴别名）；本表可填写输入按其符号补齐（如 N_hit），
    B/seqlen/kvcachelen 已由参数行引用同一输入单元格，取到的是同一个 $D$ 单元格。
    """
    table: dict = {}
    for param in group["parameters"]:
        ref = param_cells[param["id"]]
        guard = param_guards[param["id"]]
        for key in (param["symbol"], param["id"], *param.get("axis_aliases", [])):
            table.setdefault(key, (ref, guard))
    for iid, item in inputs.items():
        ref = f"$D${input_rows[iid]}"
        guard = _input_guard(item, ref)
        for key in (item["symbol"], item["id"]):
            table.setdefault(key, (ref, guard))
    return table


def _perf_compile(expr: str, table: dict, label: str) -> dict:
    """把白名单表达式编译成 Excel 片段 + 守卫 + 引用名（缺符号即具名失败）。"""
    try:
        names = expression_names(expr)
        missing = [name for name in names if name not in table]
        if missing:
            raise WorkbookSpecError(
                f"{label}: 性能规则引用了本组/本表未声明的符号 {missing}（不静默取未知）")
        refs = {name: table[name][0] for name in names}
        compiled = excel_formula(expr, refs)
    except ExpressionError as exc:
        raise WorkbookSpecError(f"{label}: 性能表达式 {expr!r} 无法编译: {exc}") from exc
    guard = _and_guard([table[name][1] for name in names])
    return {"expr": compiled, "guard": guard, "names": names, "refs": refs}


def _perf_unknown_text(names: list) -> str:
    if not names:
        return "未知（性能表达式无法求值）"
    return f"未知（依赖参数未给定或非法：{', '.join(names)}）"


def _perf_dim(ws, row: int, col: int, compiled: dict, *, fill=None,
              extra_guards: list[str] | None = None,
              reason_suffix: str = "",
              condition: dict | None = None) -> None:
    """写一维单元格：符号表达式 → 受守卫的活非负整数，否则“未知（原因）”。

    维度必须是 ≥0 的整数：负数/小数不参与成本（用户确认的 B1/M1/K64/N64 基准与
    按比例口径都只对整数规模成立）；合法零维度保留 0（如空缓存的压缩条目）。
    """
    expr = compiled["expr"]
    pieces = [compiled["guard"]]
    # 参与维度的符号本身也必须是 ≥0 的整数：避免把 H=1.5 这类小数维度乘出“看似合法”
    # 的整数收缩维（如 Nhc*H）而静默给出成本；合法零维度保留 0。
    for name in compiled["names"]:
        ref = compiled["refs"][name]
        pieces.extend([f"ISNUMBER({ref})", f"INT({ref})={ref}", f"{ref}>=0"])
    pieces.extend([f"ISNUMBER({expr})", f"INT({expr})={expr}", f"{expr}>=0"])
    pieces.extend(extra_guards or [])
    guard = f"IFERROR({_and_guard(pieces)},FALSE)"
    reason = "未知（维度须为非负整数：%s）" % ", ".join(compiled["names"])
    if reason_suffix:
        reason = reason[:-1] + "；" + reason_suffix + "）"
    if condition is not None:
        # 条件分支未执行时维度必须是 0（不依赖该分支专有尺寸），但条件本身未知/非法
        # 仍是未知：先判条件，再算维度。
        cond_reason = _perf_unknown_text(condition["names"])
        formula = (f"=IF({condition['guard']},IF({condition['expr']},"
                   f"IF({guard},{expr},{_xl_quote(reason)}),0),"
                   f"{_xl_quote(cond_reason)})")
    else:
        formula = (f"=IF({guard},{expr},{_xl_quote(reason)})")
    if len(formula) > _FORMULA_LIMIT:
        _text(ws, row, col, "（维度公式过长，未写活公式）", fill=fill)
        return
    cell = _formula(ws, row, col, formula, fill=fill)
    cell.number_format = _NUMBER_FORMAT_WORK


def _perf_row_cells(ws, row: int, rule: dict, term: dict, table: dict,
                    engine_refs: dict, *, fill=None, excluded: dict | None = None,
                    extra_note: str = "") -> dict:
    """写一行的 CUBE/VEC 维度、工作量、耗时、状态；返回单元格引用信息。"""
    kind = rule["class"]
    col_class = PERF_FIRST_COL
    col_b, col_m, col_k, col_n, col_work, col_status = (
        col_class + 1, col_class + 2, col_class + 3, col_class + 4,
        col_class + 5, col_class + 6)
    col_note = col_class + 7

    _text(ws, row, col_class, _PERF_CLASS_CN[kind], fill=fill,
          font=_FONT_LABEL)

    # 规则声明的附加约束（如 MoE 命中数与总路由分配的相容性）：不满足即未知。
    guard_compiled = [
        _perf_compile(expr, table, "guards") for expr in rule.get("guards", [])
    ]
    guard_text = rule.get("guard_text", "")
    rule_guards = [f"IFERROR(AND({item['guard']},{item['expr']}),FALSE)"
                   for item in guard_compiled]
    guard_names = sorted({name for item in guard_compiled for name in item["names"]})
    reason_suffix = guard_text or None

    conditional = rule.get("conditional")
    cond_compiled = (_perf_compile(conditional["when"], table, "conditional.when")
                     if conditional else None)

    dims: dict = {}
    if kind == "cube":
        keys = ("b", "m", "k", "n")
        cols = {"b": col_b, "m": col_m, "k": col_k, "n": col_n}
    else:
        keys = ("m", "n")
        cols = {"m": col_m, "n": col_n}
        _text(ws, row, col_b, "—", fill=fill)
        _text(ws, row, col_k, "—", fill=fill)
    for key in keys:
        dims[key] = _perf_compile(term[key], table, f"{rule.get('label', '')}/{key}")
        _perf_dim(ws, row, cols[key], dims[key], fill=fill,
                  condition=cond_compiled)
    dim_refs = [f"{get_column_letter(cols[key])}{row}" for key in keys]
    work_expr = "*".join(dim_refs)
    dims_numeric = "AND(" + ",".join(f"ISNUMBER({ref})" for ref in dim_refs) + ")"
    dim_names = sorted({name for key in keys for name in dims[key]["names"]})

    repeat = rule.get("repeat", 1)
    repeat_guard = None
    repeat_names: list = []
    if isinstance(repeat, str):
        compiled_repeat = _perf_compile(repeat, table, "repeat")
        repeat_expr = compiled_repeat["expr"]
        # 重复次数必须是 ≥1 的整数，缺失/小数/负数都不乘进工作量。
        repeat_guard = (f"IFERROR(AND({compiled_repeat['guard']},"
                        f"ISNUMBER({repeat_expr}),INT({repeat_expr})={repeat_expr},"
                        f"{repeat_expr}>=1),FALSE)")
        repeat_names = compiled_repeat["names"]
    else:
        repeat_expr = str(repeat)
    if repeat_expr != "1":
        work_expr = f"({repeat_expr})*({work_expr})"

    guard_parts = [dims_numeric, *rule_guards]
    if repeat_guard is not None:
        guard_parts.append(repeat_guard)
    work_body = f"({work_expr})"
    dim_names = sorted({*dim_names, *repeat_names, *guard_names})
    guard = _and_guard(guard_parts)
    reason = _perf_unknown_text(dim_names)
    if guard_text:
        reason = reason[:-1] + "；" + guard_text + "）"

    unknown_work = rule.get("unknown_reason") or ""
    if unknown_work:
        _text(ws, row, col_work, f"未知（{unknown_work}）", fill=_FILL_UNKNOWN)
        _text(ws, row, PERF_COST_COL, f"未知（{unknown_work}）", fill=_FILL_UNKNOWN)
        _text(ws, row, col_status, f"未知（{unknown_work}）", fill=_FILL_UNKNOWN)
    else:
        if cond_compiled is not None:
            cond_reason = _perf_unknown_text(cond_compiled["names"])
            work_formula = (
                f"=IF({cond_compiled['guard']},"
                f"IF({cond_compiled['expr']},IF({guard},{work_body},{_xl_quote(reason)}),0),"
                f"{_xl_quote(cond_reason)})")
        else:
            work_formula = f"=IF({guard},{work_body},{_xl_quote(reason)})"
        if len(work_formula) > _FORMULA_LIMIT:
            _text(ws, row, col_work, "（工作量公式过长，未写活公式）", fill=fill)
        else:
            cell = _formula(ws, row, col_work, work_formula, fill=fill)
            cell.number_format = _NUMBER_FORMAT_WORK

        thru_key = "cube" if kind == "cube" else "vec"
        thru_ref, thru_guard = engine_refs[thru_key]
        work_ref = f"{get_column_letter(col_work)}{row}"
        if excluded is not None:
            # 整网一次性项（mask/RoPE）：本页层组行只表示引用，成本在外围收一次；
            # 这里不写数值成本，避免被误加进层组小计或×层数重复收费。
            _text(ws, row, PERF_COST_COL,
                  f"整网一次性项：见 {excluded['charged_at']}（不计入本组小计）",
                  fill=_FILL_UNKNOWN)
            _text(ws, row, col_status,
                  f"引用（不重复计：整网在 {excluded['charged_at']} 收一次）",
                  fill=_FILL_UNKNOWN)
        else:
            cost_formula = (
                f"=IF(AND(ISNUMBER({work_ref}),{thru_guard}),({work_ref})/{thru_ref},"
                f"{_xl_quote(f'未知（工作量未定或 {thru_key.upper()} 吞吐非法）')})")
            if len(cost_formula) > _FORMULA_LIMIT:
                _text(ws, row, PERF_COST_COL, "（耗时公式过长，未写活公式）", fill=fill)
            else:
                cell = _formula(ws, row, PERF_COST_COL, cost_formula, fill=fill)
                cell.number_format = _NUMBER_FORMAT_COST

            if cond_compiled is not None:
                status_formula = (
                    f"=IF({cond_compiled['guard']},"
                    f"IF({cond_compiled['expr']},IF({guard},\"已计\",{_xl_quote(reason)}),"
                    f"{_xl_quote('条件不执行（计 0）：' + conditional['when_text'])}),"
                    f"{_xl_quote(cond_reason)})")
            else:
                status_formula = f"=IF({guard},\"已计\",{_xl_quote(reason)})"
            if len(status_formula) > _FORMULA_LIMIT:
                _text(ws, row, col_status, "（状态公式过长）", fill=fill)
            else:
                _formula(ws, row, col_status, status_formula, fill=fill)

    note = rule.get("note", "")
    suffix = []
    if rule.get("repeat") not in (None, 1, "1"):
        suffix.append(f"同形重复 {rule['repeat']} 次（工作量已乘）")
    if conditional:
        suffix.append("条件执行：" + conditional["when_text"])
    if guard_text:
        suffix.append(guard_text)
    if excluded is not None:
        suffix.append(excluded["reason"])
    if extra_note:
        suffix.append(extra_note)
    note_text = (note + ("；" + "；".join(suffix) if suffix else "")).strip()
    _text(ws, row, col_note, note_text, fill=fill)
    return {"work": f"{get_column_letter(col_work)}{row}",
            "cost": (None if excluded is not None
                     else f"{get_column_letter(PERF_COST_COL)}{row}")}


# ---------------------------------------------------------------- 渲染


def _render_sheet_summary(ws, spec: dict, sheet: dict, layout: dict) -> int:
    """工作表横幅与整表说明（模型、场景、性能公式状态、呈现约定）。"""
    model = spec["model"]
    performance = spec["performance"]
    engine = performance.get("engine")
    _banner(ws, 1, f"{model['display_name']} · {sheet['title']}", _FILL_TITLE, big=True,
            n_cols=layout["n_cols"])
    perf_text = performance["status"]
    if performance["note"]:
        perf_text += f"；{performance['note']}"
    facts = [
        ("工作表", f"{sheet['name']}（id={sheet['id']}）"),
        ("模型", f"{model['display_name']}（{model['model_key']}）"),
        ("实现", f"{model['implementation']} · source_revision {model['source_revision']}"),
        ("配置文件", model["config_path"]),
        ("场景说明", " | ".join(sheet["notes"]) or "（未提供）"),
        ("性能公式", perf_text),
    ]
    if engine is not None:
        facts.append((
            "性能区读法",
            ("逐算子：分类（CUBE/VEC）→ B/M/K/N 或 M/N（活公式，引用本组参数与本表输入）"
             "→ 工作量 → 用户规则耗时 ms → 成本状态；层组末尾给单层小计与"
             "层组总计（单层小计×真实层数），整网末尾按串行顺序求和各层组总计与"
             "入口/出口外围小计，父小计不再重复计入。未知/非法输入显示“未知（原因）”，"
             "不按 0 计；只有全部必要项已定时整网总计才是数值，否则给已知部分小计与缺口数。")))
    facts.append(("呈现约定", spec["disclaimer"]))
    for offset, (label, value) in enumerate(facts):
        row = 2 + offset
        _text(ws, row, 1, label, font=_FONT_LABEL, alignment=_TOP, fill=_FILL_LABEL, border=False)
        _text(ws, row, 2, value, alignment=_TOP, border=False)
    return 2 + len(facts) + 1  # 空一行分隔下一区


def _input_kind_label(item: dict) -> str:
    if item["value"] is None:
        return "未知（未给定）"
    return {
        "integer": "整数输入", "number": "数值输入",
        "text": "文本输入", "boolean": "布尔输入",
    }[item["kind"]]


def _render_inputs(ws, start_row: int, sheet: dict, layout: dict):
    """可填写的性能配置区。返回 (下一行, 配置id→行号, 配置id→配置项)。"""
    inputs = sheet["inputs"]
    _banner(ws, start_row,
            f"性能配置与场景输入（{len(inputs)} 项；黄色单元格可填写，修改后由本表公式重算；"
            f"示例值不是用户给定负载）", _FILL_PARAM_SECTION, n_cols=layout["n_cols"])
    head_row = start_row + 1
    _header_row(ws, head_row, INPUT_COLUMNS, _FILL_PARAM_HEAD)
    row_of_input: dict = {}
    by_id: dict = {}
    row = head_row + 1
    for item in inputs:
        row_of_input[item["id"]] = row
        by_id[item["id"]] = item
        _text(ws, row, 1, item["id"])
        _text(ws, row, 2, item["symbol"], font=_FONT_LABEL)
        _text(ws, row, 3, item["label"])
        value = item["value"]
        if isinstance(value, bool) or isinstance(value, (int, float)):
            cell = _number(ws, row, 4, value)
            if isinstance(value, bool):
                cell.data_type = "b"
        elif value is None:
            _text(ws, row, 4, "未知（未给定）", fill=_FILL_UNKNOWN)
        else:
            _text(ws, row, 4, _canonical(value))
        if item["editable"]:
            ws.cell(row=row, column=4).fill = _FILL_INPUT
        _text(ws, row, 5, _input_kind_label(item))
        _text(ws, row, 6, item["source"])
        _text(ws, row, 7, "可改" if item["editable"] else "只读")
        _text(ws, row, 8, "本表输入")
        _text(ws, row, 9, item["note"])
        row += 1
    if not inputs:
        _text(ws, row, 1, "（本表未声明可填写配置）", font=_FONT_LABEL)
        row += 1
    return row + 1, row_of_input, by_id


def _render_group_summary(ws, start_row: int, group: dict, layout: dict,
                          engine: dict | None) -> int:
    layers = ", ".join(str(index) for index in group["layer_indices"])
    representative = group["representative_layer_index"]
    periphery = group.get("kind") == "periphery"
    if periphery:
        layer_banner = "整网外围（不属于 Decoder 层组）"
        layer_fact = ("（整网外围；不占真实层号，层内结构仍由四个层组表达）")
    elif layers:
        layer_banner = f"真实层 [{layers}] · 代表层 {representative}"
        layer_fact = (f"[{layers}]（{len(group['layer_indices'])} 层）；代表层 {representative}")
    else:
        layer_banner = "未声明 Decoder 层号（通用单页分组）"
        layer_fact = "（未声明；通用单页不伪造层号，DeepSeek 适配必须提供）"
    loop = group["loop"]
    loop_text = ("（未声明循环归属）" if loop is None else
                 f"{loop['label']} × {loop['count']}；{loop['condition']}")
    _banner(ws, start_row,
            f"层组 {group['label']}（{group['id']}） · {layer_banner}",
            _FILL_PARAM_SECTION, n_cols=layout["n_cols"])
    if engine is None:
        cost_fact = "待提供（用户性能公式/XPU 未提供；本表不填猜测耗时，不记 0）"
    elif periphery:
        cost_fact = ("按用户 CUBE/VEC 规则写入本块算子行；入口/出口外围分别给"
                     "小计（不乘层数），层循环成本不在此块重复计。")
    else:
        cost_fact = ("按用户 CUBE/VEC 规则写入本组算子行；本组末尾给单层小计与"
                     "层组总计（单层小计×本组真实层数），整网只引用层组总计。")
    facts = [
        ("层组类型", f"attention_type={group['attention_type']}；moe_type={group['moe_type']}"),
        ("真实层索引", layer_fact),
        ("循环归属", loop_text),
        ("模板复用", group["template_note"]),
        ("本表范围", group["scope_note"]),
        ("缓存说明", " | ".join(group["cache_notes"]) or "（未提供）"),
        ("覆盖说明", " | ".join(group["coverage_notes"]) or "（未提供）"),
        ("性能成本", cost_fact),
    ]
    for offset, (label, value) in enumerate(facts):
        row = start_row + 1 + offset
        _text(ws, row, 1, label, font=_FONT_LABEL, alignment=_TOP, fill=_FILL_LABEL, border=False)
        _text(ws, row, 2, value, alignment=_TOP, border=False)
    return start_row + 1 + len(facts) + 1


def _unknown(reason: str) -> dict:
    return {"kind": "unknown", "known": False, "reason": reason, "numeric": False}


def _and_guard(guards: list[str]) -> str:
    unique = [guard for index, guard in enumerate(guards) if guard not in guards[:index]]
    return f"AND({','.join(unique)})" if unique else "TRUE"


def _input_guard(item: dict, ref: str) -> str:
    """输入单元格的活校验：空白/文本/小数/负数都不当作 0 或有效值。"""
    kind = item["kind"]
    if kind == "integer":
        # Excel AND 不短路：文本 "abc" 会让 INT(#VALUE!) 先报错，必须整体兜住。
        return f"IFERROR(AND(ISNUMBER({ref}),INT({ref})={ref},{ref}>=0),FALSE)"
    if kind == "number":
        return f"ISNUMBER({ref})"
    if kind == "boolean":
        return f"ISLOGICAL({ref})"
    return f"AND(ISTEXT({ref}),LEN({ref})>0)"


def _domain_guard(param: dict, ref: str, symbol_cells: dict,
                  symbol_guards: dict) -> str | None:
    """可选参数有效域守卫（整数/下界/上界符号），不改变其它模型既有行。"""
    domain = param.get("domain")
    if not domain:
        return None
    if domain.get("kind") != "integer":
        return "FALSE"
    parts = [f"ISNUMBER({ref})", f"INT({ref})={ref}"]
    minimum = domain.get("min")
    if minimum is not None:
        parts.append(f"{ref}>={minimum}")
    max_symbol = domain.get("max_symbol")
    if max_symbol:
        max_ref = symbol_cells.get(max_symbol)
        max_guard = symbol_guards.get(max_symbol)
        if not max_ref or not max_guard:
            return "FALSE"
        parts.append(f"{ref}<{max_ref}")
        parts.append(max_guard)
    return f"IFERROR(AND({','.join(parts)}),FALSE)"


def _resolved_value(param: dict, symbol_rows: dict, symbol_cells: dict,
                    symbol_guards: dict, input_rows: dict, inputs: dict,
                    own_guard: str | None = None) -> dict:
    """解析一个参数：字面量 / 引用本表输入 / **活公式** / 结构性未知。

    公式不在导出时把未知折成静态文本：派生式与输入引用都写成带 IF 守卫的
    Excel 公式。输入空白/非法时显示“未知（原因）”，用户恢复或补填后由 Excel
    重新得到数值——这是同一份现成工作簿的编辑链，不依赖重新导出。
    """
    formula = param.get("formula")
    if formula is not None:
        try:
            names = expression_names(formula)
        except ExpressionError as exc:
            return _unknown(f"表达式非法: {exc}")
        missing = [name for name in names if name not in symbol_rows]
        if missing:
            return _unknown("缺少参数: " + ", ".join(missing))
        ambiguous = [name for name in names if len(symbol_rows[name]) > 1]
        if ambiguous:
            return _unknown("符号存在歧义: " + ", ".join(ambiguous))
        refs = {name: symbol_cells[name] for name in names}
        try:
            compiled = excel_formula(formula, refs)
        except ExpressionError as exc:
            return _unknown(f"表达式无法转 Excel: {exc}")
        guards = [symbol_guards[name] for name in names]
        if own_guard:
            guards.append(own_guard)
        guard = _and_guard(guards)
        unknown_reason = (
            f"未知（依赖参数未给定或非法：{', '.join(names)}）"
            if names else "未知（表达式求值失败）")
        fail_reason = f"未知（公式求值失败：{formula}）"
        wrapped = (f"=IF({guard},IFERROR({compiled},{_xl_quote(fail_reason)}),"
                   f"{_xl_quote(unknown_reason)})")
        if len(wrapped) > _FORMULA_LIMIT:
            return _unknown("Excel 公式过长，未写入可重算公式")
        return {"kind": "formula", "known": True, "numeric": True,
                "value": None, "formula": wrapped}

    value_ref = param.get("value_ref")
    if value_ref is not None:
        item = inputs.get(value_ref)
        if item is None:
            return _unknown(f"引用输入 {value_ref} 不存在")
        ref = f"$D${input_rows[value_ref]}"
        base_guard = _input_guard(item, ref)
        if own_guard:
            guard = f"AND({base_guard},{own_guard})"
            reason = (f"未知（输入 {value_ref} 不满足约束："
                      f"{param.get('consistency', '一致性')}）")
        else:
            guard = base_guard
            reason = f"未知（输入 {value_ref} 未给定或非法）"
        formula = f"=IF({guard},{ref},{_xl_quote(reason)})"
        if len(formula) > _FORMULA_LIMIT:
            return _unknown("Excel 公式过长，未写入可重算公式")
        numeric = item["kind"] in ("integer", "number")
        return {"kind": "ref", "known": True, "numeric": numeric,
                "value": item["value"], "formula": formula}

    reason = param.get("unknown_reason")
    if reason:
        return _unknown(reason)

    value = param.get("value")
    if value is None:
        return {"kind": "null", "known": False, "numeric": False, "value": None}
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    return {"kind": "literal", "known": True, "numeric": numeric, "value": value}


def _resolve_parameter_values(params: list, row_of_id: dict, input_rows: dict,
                              inputs: dict) -> tuple[dict, dict, dict]:
    """解析全部参数并返回符号→单元格/守卫映射（公式直接引用这些活单元格）。"""
    symbol_rows: dict = {}
    symbol_cells: dict = {}
    symbol_guards: dict = {}
    param_cells: dict = {}
    param_guards: dict = {}
    for param in params:
        symbol_rows.setdefault(param["symbol"], []).append(row_of_id[param["id"]])
        # 参数 id 也作为可引用别名（结构/派生式里 id 与显示符号不同的模型，如
        # Qwen 的 Dqkv/D_qkv、Pfrac/partial_rotary_factor）；同一 id 与其它符号
        # 冲突时由下方 ambiguous 检查具名拒绝，不静默选一个。
        if param["id"] != param["symbol"]:
            symbol_rows.setdefault(param["id"], []).append(row_of_id[param["id"]])
        value_ref = param.get("value_ref")
        if value_ref in input_rows:
            ref = f"$D${input_rows[value_ref]}"
            item = inputs.get(value_ref)
            guard = _input_guard(item, ref) if item is not None else "FALSE"
        else:
            ref = f"$D${row_of_id[param['id']]}"
            guard = f"ISNUMBER({ref})"
        domain_guard = _domain_guard(param, ref, symbol_cells, symbol_guards)
        if domain_guard is not None:
            guard = domain_guard
        # 每个参数自己的单元格/守卫（按 id 记录，避免同名符号互相覆盖；算子实际
        # shape 必须用这一份，而不是按符号取最后一个）。
        param_cells[param["id"]] = ref
        param_guards[param["id"]] = guard
        symbol_cells[param["symbol"]] = ref
        symbol_guards[param["symbol"]] = guard
        for alias in param.get("axis_aliases", []):
            symbol_cells[alias] = ref
            symbol_guards[alias] = guard
    # 可选一致性约束（白名单布尔表达式）：只有满足约束的输入才允许被下游
    # shape/公式引用；不满足时依赖量显示“未知（不满足约束）”，恢复后重算。
    # 旧模型没有 consistency 键 → 这里为空，输出与既有工作簿逐字节一致。
    consistency_guards: dict[str, str] = {}
    for param in params:
        expr = param.get("consistency")
        if not expr:
            continue
        names = expression_names(expr)
        missing = [name for name in names if name not in symbol_cells]
        if missing:
            raise WorkbookSpecError(
                f"参数 {param['id']} 的 consistency 引用未声明符号: {', '.join(missing)}")
        refs = {name: symbol_cells[name] for name in names}
        compiled = excel_formula(expr, refs)
        dep_guard = _and_guard([symbol_guards[name] for name in names])
        # Excel 的 AND 不短路：文本/空白参与算术会先产生 #VALUE!，必须整体兜住。
        guard = f"AND({dep_guard},IFERROR(({compiled}),FALSE))"
        consistency_guards[param["id"]] = guard
        param_guards[param["id"]] = f"AND({param_guards[param['id']]},{guard})"
        for key in (param["symbol"], *param.get("axis_aliases", [])):
            symbol_guards[key] = f"AND({symbol_guards[key]},{guard})"
    resolved = {
        param["id"]: _resolved_value(
            param, symbol_rows, symbol_cells, symbol_guards, input_rows, inputs,
            own_guard=consistency_guards.get(param["id"]))
        for param in params
    }
    return resolved, symbol_cells, symbol_guards, param_cells, param_guards


def _render_parameters(ws, start_row: int, group: dict, sheet_name: str,
                       input_rows: dict, inputs: dict, layout: dict):
    """参数符号区（可填写输入/派生公式/固定配置）。返回行号与本组索引。"""
    params = group["parameters"]
    _banner(ws, start_row,
            f"参数符号区（{len(params)} 项；算子与张量的 parameter_refs 只引用本组，"
            f"派生值由 Excel 公式重算）", _FILL_PARAM_SECTION, n_cols=layout["n_cols"])
    head_row = start_row + 1
    _header_row(ws, head_row, PARAM_COLUMNS, _FILL_PARAM_HEAD)

    row = head_row + 1
    row_of_id = {param["id"]: row + offset for offset, param in enumerate(params)}
    resolved, symbol_cells, symbol_guards, param_cells, param_guards = \
        _resolve_parameter_values(params, row_of_id, input_rows, inputs)

    rows_by_symbol: dict = {}
    id_of_row: dict = {}
    for param in params:
        current = row_of_id[param["id"]]
        id_of_row[current] = param["id"]
        rows_by_symbol.setdefault(param["symbol"], []).append(current)
        for alias in param.get("axis_aliases", []):
            if current not in rows_by_symbol.setdefault(alias, []):
                rows_by_symbol[alias].append(current)
        _text(ws, current, 1, param["id"])
        _text(ws, current, 2, param["symbol"], font=_FONT_LABEL)
        _text(ws, current, 3, param["meaning"])
        outcome = resolved[param["id"]]
        if outcome["kind"] == "literal" and isinstance(outcome["value"], bool):
            cell = _number(ws, current, 4, int(outcome["value"]))
            cell.data_type = "b"
        elif outcome["kind"] == "literal" and isinstance(outcome["value"], (int, float)):
            _number(ws, current, 4, outcome["value"])
        elif outcome["kind"] in ("formula", "ref"):
            _formula(ws, current, 4, outcome["formula"], fill=_FILL_FORMULA)
        elif outcome["kind"] == "null":
            _text(ws, current, 4, "null", fill=_FILL_UNKNOWN)
        elif outcome["kind"] == "literal":
            _text(ws, current, 4, _canonical(outcome["value"]))
        else:
            _text(ws, current, 4, f"未知（{outcome['reason']}）", fill=_FILL_UNKNOWN)
        if outcome["kind"] == "ref":
            kind_text = "输入（本表配置）"
        elif outcome["kind"] == "formula":
            kind_text = "派生（Excel 公式）"
        elif outcome["kind"] == "unknown":
            kind_text = "未知"
        elif outcome["kind"] == "null":
            kind_text = "null"
        else:
            kind_text = _value_kind(outcome["value"])
        _text(ws, current, 5, kind_text)
        _text(ws, current, 6, param["expression"] or "（无表达式）")
        _text(ws, current, 7, ", ".join(sorted({r["kind"] for r in param["source_refs"]}))
              or "（未提供）")
        _text(ws, current, 8, _join_sources(param["source_refs"]))
        _text(ws, current, 9, param["note"])
    if params:
        row = head_row + 1 + len(params)
    if not params:
        _text(ws, row, 1, "（本阶段文档未提供参数）", font=_FONT_LABEL)
        row += 1
    return (row + 1, row_of_id, rows_by_symbol, id_of_row, symbol_cells,
            symbol_guards, param_cells, param_guards)


def _is_symbolic_axis(axis) -> bool:
    """只有真正的符号/表达式轴才参与参数区联查；纯数字字面量（含字符串形式）是具体尺度。"""
    return isinstance(axis, str) and not re.fullmatch(r"-?\d+", axis.strip())


# 轻量复合轴识别：只求“哪些已声明参数参进了这条轴”，不求值、不建通用求值器。
_AXIS_BAD_CHAR_RE = re.compile(r"[^A-Za-z0-9_. ×*/÷+(),-]")
_AXIS_RUN_RE = re.compile(
    r"\d+(?:\.\d+)?[A-Za-z_][A-Za-z0-9_]*"   # 无乘号系数写法：2Dh、16H（当一个 token 读）
    r"|[A-Za-z_][A-Za-z0-9_]*"
    r"|\d+(?:\.\d+)?")                         # 纯常数
_AXIS_COEF_RE = re.compile(r"^(\d+(?:\.\d+)?)([A-Za-z_][A-Za-z0-9_]*)$")
_AXIS_GAP_CHARS = set("×*/÷+-(), \t")
_AXIS_FUNCTIONS = ("min", "max")


def _classify_axis(axis: str, defined_symbols: set) -> tuple:
    """把一条符号轴分为直接符号 / 可轻量分解的复合轴 / 不认识的写法。

    返回 (kind, 组成符号列表, 原轴文本)：
    - "simple"：轴文本本身就是一个已声明符号（输出与原行为逐字一致）。
    - "composite"：由常数、×/*、÷/、+/-、括号、min/max 与已声明符号组成；
      已声明符号按 token 边界精确匹配（绝不把 Nh 拆成 N+h，也绝不把 H 当作 Nh 的子串），
      “常数系数+符号”（2Dh/2Es）当一个 token 识别为系数与已声明符号。
      未声明的标识符列入未命中；函数名与常数不算缺失参数。
    - "opaque"：写法超出支持范围（如 ⌈T/2⌉、sqrt(...)、A B 并列、××）→原样保留并提示。
    """
    text = axis.strip()
    if not text:
        return "opaque", [], axis
    if text in defined_symbols:
        return "simple", [text], axis
    if _AXIS_BAD_CHAR_RE.search(text) or text.count("(") != text.count(")"):
        return "opaque", [], axis

    matches = list(_AXIS_RUN_RE.finditer(text))
    if not matches:
        return "opaque", [], axis

    components: list = []
    unknown: list = []
    position = 0
    seen_operand = False
    for match in matches:
        gap = text[position:match.start()]
        position = match.end()
        run = match.group(0)
        if any(ch not in _AXIS_GAP_CHARS for ch in gap):        # 如 "2.h"、“config.h”
            return "opaque", [], axis
        operators = [ch for ch in gap if ch in "×*/÷+-"]
        if seen_operand and not gap.strip():                    # 操作数并列（"B T"）：不猜乘号
            return "opaque", [], axis
        if len(operators) > 1:                                  # 运算符相邻（"××"、"+-"）
            return "opaque", [], axis
        seen_operand = True
        after = text[position:].lstrip()
        if run in defined_symbols:
            if run not in components:
                components.append(run)
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", run):                 # 常数：不是缺失参数
            continue
        if run in _AXIS_FUNCTIONS and after.startswith("("):    # 函数名：不是缺失参数
            continue
        coef = _AXIS_COEF_RE.match(run)
        if coef and coef.group(2) in defined_symbols:           # 2Dh → 系数 2 + 已声明符号 Dh
            if coef.group(2) not in components:
                components.append(coef.group(2))
            continue
        if after.startswith("("):                               # 不认识的函数调用
            return "opaque", [], axis
        if run not in unknown:
            unknown.append(run)
    return "composite", components + unknown, axis


def _axis_units(operator: dict, tensor_index: dict, defined_symbols: set) -> list:
    """按轴出现顺序生成待联查单元；每条单元记住自己的原文与组成符号。"""
    units: list = []
    seen: list = []
    for tid in (*operator["inputs"], *operator["outputs"], *operator["weights"]):
        for axis in tensor_index[tid]["shape"]:
            if not _is_symbolic_axis(axis) or axis in seen:
                continue
            seen.append(axis)
            kind, symbols, _text = _classify_axis(axis, defined_symbols)
            units.append({"axis": axis, "kind": kind, "symbols": symbols, "rows": []})
    return units


def _operator_param_ids(operator: dict, tensor_index: dict) -> list:
    """算子显式引用 + 所用张量显式引用（去重保序，只用文档声明的引用）。"""
    involved = [tensor_index[tid] for tid in
                (*operator["inputs"], *operator["outputs"], *operator["weights"])]
    return _dedup([*operator["parameter_refs"],
                   *(pid for tensor in involved for pid in tensor["parameter_refs"])])


def _bind_axis_symbols(symbols: list, rows_by_symbol: dict, id_of_row: dict,
                       explicit_ids: set):
    """用显式 parameter_refs 定位轴符号对应的本表参数行，同名字母不能默认取第一个。

    返回 (已绑定行, 歧义符号及其同名候选, 本表无同名参数的符号)：
    - 同名候选只有一个 → 直接绑定（原有行为）。
    - 多个同名候选 → 只在当前行（算子及其输入/输出/权重张量）的显式引用能唯一分辨时绑定。
    - 显式引用分辨不出（零个或多个候选被引用）→ 记为歧义，不猜值、不静静选错。
    """
    bound: list = []
    ambiguous: list = []
    missed: list = []
    for symbol in symbols:
        candidates = rows_by_symbol.get(symbol) or []
        if not candidates:
            missed.append(symbol)
            continue
        if len(candidates) == 1:
            bound.append(candidates[0])
            continue
        hits = [r for r in candidates if id_of_row[r] in explicit_ids]
        if len(hits) == 1:
            bound.append(hits[0])
        else:
            ambiguous.append((symbol, [(id_of_row[r], r) for r in (hits or candidates)]))
    return bound, ambiguous, missed


def _unbound_note(ambiguous: list, missed: list, unsupported: list | None = None) -> str | None:
    """歧义/未命中/不支持写法的中文陈述：给出相关参数id与本表行号，或原轴文本。"""
    parts = []
    if ambiguous:
        bodies = [f"{symbol}（本表同名候选："
                  + "、".join(f"{pid}（第{r}行）" for pid, r in pairs) + "）"
                  for symbol, pairs in ambiguous]
        parts.append("符号存在歧义，未取配置值：" + "；".join(bodies))
    if missed:
        parts.append("未命中本表参数区同名符号：" + ", ".join(missed))
    if unsupported:
        parts.append("原样保留（超出轻量表达式支持，未求值）：" + "、".join(unsupported))
    return "；".join(parts) if parts else None


def _bind_axis_units(units: list, rows_by_symbol: dict, id_of_row: dict, explicit_ids: set):
    """逐轴单元绑定；同名候选仍由本行显式 parameter_refs 分辨，分不出就不取。

    返回 (歧义符号集, 未命中符号集, 不支持的原轴文本)；绑定行写回各单元的 "rows"。
    复合轴只用该轴自己出现的 token，不把无关 parameter_refs 伪装成轴组成符号。
    """
    ambiguous_all: list = []
    missed_all: list = []
    for unit in units:
        if unit["kind"] == "opaque":
            continue
        bound, ambiguous, missed = _bind_axis_symbols(
            unit["symbols"], rows_by_symbol, id_of_row, explicit_ids)
        unit["rows"] = _dedup(bound)
        for symbol, pairs in ambiguous:
            if symbol not in [s for s, _ in ambiguous_all]:
                ambiguous_all.append((symbol, pairs))
        for symbol in missed:
            if symbol not in missed_all:
                missed_all.append(symbol)
    unsupported = [u["axis"] for u in units if u["kind"] == "opaque"]
    return ambiguous_all, missed_all, unsupported


def _axis_lookup_formula(units: list, note: str | None = None,
                         labeled: bool = True) -> str | None:
    """「维度符号→本表配置值」列公式：每个组成符号单独链接本表参数行单元格。

    直接符号保持原有简洁形式；复合轴加“原轴文本组成：”字面量前缀，只列组成参数，
    不对轴求值。无绑定或公式过长时返回 None（调用方回退为文本）。
    """
    pieces: list = []
    shown = 0
    overflow = 0
    for unit in units:
        rows = unit["rows"]
        if not rows:
            continue
        if shown + len(rows) > _MAX_INLINE_REFS:
            overflow += len(rows)
            continue
        parts = [f'$B${r}&"="&$D${r}' for r in rows]
        body = '&"; "&'.join(parts)
        if labeled and unit["kind"] == "composite":
            # 复合轴：先写字面量标签“原轴文本组成：”，再直接接上该轴的组成符号。
            pieces.append(_xl_quote(f"{unit['axis']}组成：") + "&" + body)
        else:
            pieces.append(body)
        shown += len(rows)
    if not pieces:
        return None
    formula = "=" + ' & "; " & '.join(pieces)
    if overflow:
        formula += f' & "; " & {_xl_quote(f"…另有{overflow}项，完整清单见「参数引用」列")}'
    if note:
        formula += f' & "; " & {_xl_quote(note)}'
    if len(formula) > _FORMULA_LIMIT:
        return None
    return formula


def _normalize_axis_expression(axis: str) -> str:
    """把结构 shape 的展示写法转成白名单表达式（只做语法归一，不求值）。"""
    text = str(axis).strip().replace("×", "*").replace("÷", "//")
    # 2Dh / 2Es / 16H：数字系数与符号之间补乘号（只处理数字紧跟标识符）。
    return re.sub(r"(?<=\d)(?=[A-Za-z_])", "*", text)


def _axis_symbol_row(name: str, rows_by_symbol: dict, id_of_row: dict,
                     explicit_ids: set):
    """按既有显式 parameter_refs 权威为一个轴符号选行；分不出就是歧义。"""
    candidates = rows_by_symbol.get(name) or []
    if not candidates:
        return None, f"未命中参数: {name}"
    if len(candidates) == 1:
        return candidates[0], None
    hits = [row for row in candidates if id_of_row[row] in explicit_ids]
    if len(hits) == 1:
        return hits[0], None
    return None, f"符号存在歧义: {name}"


def _compile_axis(axis, row_cells: dict, row_guards: dict, rows_by_symbol: dict,
                  id_of_row: dict, explicit_ids: set) -> dict:
    """编译单个轴：返回 expr/guards/names 或带原因的 error。

    同名符号沿用算子/张量显式 parameter_refs 消歧；零个或多个明确候选才未知。
    """
    text = str(axis).strip()
    if text in rows_by_symbol:
        row, error = _axis_symbol_row(text, rows_by_symbol, id_of_row, explicit_ids)
        if error:
            return {"error": f"轴 {text!r} {error}"}
        return {"expr": row_cells[row], "guards": [row_guards[row]], "names": [text]}
    if re.fullmatch(r"-?\d+", text):
        return {"expr": text, "guards": [], "names": []}
    normalized = _normalize_axis_expression(text)
    try:
        names = expression_names(normalized)
    except ExpressionError as exc:
        return {"error": f"轴 {text!r} 超出白名单: {exc}"}
    refs: dict = {}
    guards: list = []
    for name in names:
        row, error = _axis_symbol_row(name, rows_by_symbol, id_of_row, explicit_ids)
        if error:
            return {"error": f"轴 {text!r} {error}"}
        refs[name] = row_cells[row]
        guards.append(row_guards[row])
    try:
        expr = excel_formula(normalized, refs)
    except ExpressionError as exc:
        return {"error": f"轴 {text!r} 无法转 Excel: {exc}"}
    return {"expr": expr, "guards": guards, "names": names}


def _actual_shape_expr(tensor: dict, row_cells: dict, row_guards: dict,
                       rows_by_symbol: dict, id_of_row: dict,
                       explicit_ids: set) -> str:
    """张量的实际 shape 文本表达式（无前导 =；未知/不支持带原因，不猜值）。"""
    parts, guards, names = [], [], []
    for axis in tensor["shape"]:
        compiled = _compile_axis(
            axis, row_cells, row_guards, rows_by_symbol, id_of_row, explicit_ids)
        if "error" in compiled:
            return _xl_quote(f"未知（{compiled['error']}）")
        parts.append(compiled["expr"])
        guards.extend(compiled["guards"])
        names.extend(compiled["names"])
    if not parts:
        return '"[]"'
    guard = _and_guard(guards)
    shape_expr = '"["&' + '&","&'.join(parts) + '&"]"'
    reason = ("未知（依赖参数未给定或非法："
              + ", ".join(sorted(set(names))) + "）") if names else "未知（shape 求值失败）"
    return f"IF({guard},{shape_expr},{_xl_quote(reason)})"


def _actual_tensors_expr(ids: list, tensor_index: dict, row_cells: dict,
                         row_guards: dict, rows_by_symbol: dict, id_of_row: dict,
                         explicit_ids: set) -> str:
    if not ids:
        return _xl_quote("（无）")
    parts = []
    for tid in ids:
        tensor = tensor_index[tid]
        shape_expr = _actual_shape_expr(
            tensor, row_cells, row_guards, rows_by_symbol, id_of_row, explicit_ids)
        parts.append(_xl_quote(f"{tensor['name']}=") + "&(" + shape_expr + ")")
    return '&"; "&'.join(parts)


def performance_status(spec: dict) -> str:
    """性能区状态文案（未启用性能区时沿用既有“待提供”契约）。"""
    return spec["performance"]["status"]


def _engine_throughput_refs(engine: dict, input_rows: dict, inputs: dict) -> dict:
    """CUBE/VEC 吞吐引用（本表可改单元格，>0 才有效）。"""
    refs = {}
    for key, input_id in (("cube", engine["cube_throughput_input"]),
                          ("vec", engine["vec_throughput_input"])):
        ref = f"$D${input_rows[input_id]}"
        refs[key] = (ref, f"AND(ISNUMBER({ref}),{ref}>0)")
    return refs


def _render_container_row(ws, row: int, rule: dict, input_rows: dict, inputs: dict,
                          layout: dict, fill=None) -> None:
    """容器节点（如层循环）：类别明确、成本 0 并写明原因，不冒充算子耗时。"""
    col = PERF_FIRST_COL
    _text(ws, row, col, _PERF_CLASS_CN["container"], fill=fill, font=_FONT_LABEL)
    for offset in (1, 2, 3, 4):
        _text(ws, row, col + offset, "—", fill=fill)
    work = _number(ws, row, col + 5, 0)
    work.number_format = _NUMBER_FORMAT_WORK
    cost = _number(ws, row, PERF_COST_COL, 0)
    cost.number_format = _NUMBER_FORMAT_COST
    _text(ws, row, col + 6, f"容器（不计成本）：{rule['reason']}", fill=fill)
    _text(ws, row, col + 7, rule["note"], fill=fill)


def _render_operators(ws, start_row: int, group: dict, spec: dict, layout: dict,
                      input_rows: dict, inputs: dict, row_of_id: dict,
                      rows_by_symbol: dict, id_of_row: dict, param_cells: dict,
                      param_guards: dict, sheet_operator_ids: set) -> dict:
    engine = spec["performance"].get("engine")
    sheet_name = group["sheet_name"]
    tensor_index = group["tensor_index"]
    # 实际 shape 使用与参数区**同一份**权威单元格/守卫（含 domain/consistency，
    # 按参数 id 记录以避免同名符号互相覆盖），不在这里另建简化版：否则参数区显示
    # 未知、算子 Q/R/S 列仍显示有效尺寸。
    row_cells: dict = {}
    row_guards: dict = {}
    for param in group["parameters"]:
        current = row_of_id[param["id"]]
        row_cells[current] = param_cells[param["id"]]
        row_guards[current] = param_guards[param["id"]]
    _banner(ws, start_row,
            f"算子区（{len(group['operators'])} 行，按实际阅读顺序；"
            f"◆ 黑盒 = 内部未分析，参数/维度联查列指向本组参数区）", _FILL_OP_SECTION,
            n_cols=layout["n_cols"])
    head_row = start_row + 1
    _header_row(ws, head_row, layout["columns"], _FILL_OP_HEAD)
    layers = ", ".join(str(index) for index in group["layer_indices"])
    loop = group["loop"]
    if loop is None:
        loop_text = "（未声明循环归属）"
    else:
        layer_part = f"真实层 [{layers}] · " if layers else "（未声明层号） · "
        loop_text = (f"{group['label']} · {layer_part}×{loop['count']}；"
                     f"{loop['condition']}")
    cost_text = f"{performance_status(spec)}；不填猜测耗时、不把缺失成本记 0"

    perf_table = (_perf_symbol_table(group, param_cells, param_guards, input_rows, inputs)
                  if engine is not None else None)
    engine_refs = (_engine_throughput_refs(engine, input_rows, inputs)
                   if engine is not None else None)

    op_rows: dict = {}
    charged_rows: list[int] = []
    excluded_rows: list[int] = []
    row = head_row + 1
    for op in group["operators"]:
        blackbox = op["analysis_status"] == "internal_unanalyzed"
        row_fill = _FILL_BLACK_ROW if blackbox else None
        status_font = _FONT_BLACK if blackbox else _FONT_BODY
        first_row = row

        def put(col: int, value, font=None, _row=row):
            return _text(ws, _row, col, value,
                         font=font or (status_font if blackbox and col in (1, 13) else _FONT_BODY),
                         fill=row_fill)

        put(1, "◆ 黑盒（内部未分析）" if blackbox else "已展开")
        put(2, op["id"])
        put(3, op["module_path"])
        put(4, op["name"])
        put(5, op["op"])

        def group_tensors(ids: list) -> str:
            return (", ".join(_tensor_label(tensor_index[t]) for t in ids)
                    if ids else "（无）")

        put(6, group_tensors(op["inputs"]))
        put(7, group_tensors(op["outputs"]))
        put(8, group_tensors(op["weights"]))

        # 显式引用集是本组绑定权威：先算出来，再拿它分辨轴符号。
        ref_ids = _operator_param_ids(op, tensor_index)
        ref_rows = [row_of_id[pid] for pid in ref_ids]

        units = _axis_units(op, tensor_index, set(rows_by_symbol))
        ambiguous, missed, unsupported = _bind_axis_units(
            units, rows_by_symbol, id_of_row, set(ref_ids))
        note = _unbound_note(ambiguous, missed, unsupported)
        formula = _axis_lookup_formula(units, note)
        if formula is None:
            formula = _axis_lookup_formula(units)          # 附注过长：只保留联查
        if formula is None:
            formula = _axis_lookup_formula(units, labeled=False)  # 标签过长：回到无标签形式
        if formula:
            _formula(ws, row, 9, formula, fill=row_fill)
        else:
            put(9, note if note else "（无符号轴）")

        link = put(10, ", ".join(ref_ids) if ref_ids else "（无参数引用）")
        if ref_ids:
            target = row_of_id[ref_ids[0]]
            link.hyperlink = Hyperlink(
                ref=link.coordinate,
                location=f"'{sheet_name}'!$A${target}",
                tooltip=f"定位本组参数区 {ref_ids[0]}（第 {target} 行）",
                display=link.value,
            )
            link.font = _FONT_LINK
        # 引用的参数中存在同名符号时，联查附id（$A$ 同行），否则保持原有简洁形式。
        ref_symbols = [group["param_index"][pid]["symbol"] for pid in ref_ids]
        dup_symbols = {s for s in ref_symbols if ref_symbols.count(s) > 1}
        k_ids = [pid if s in dup_symbols else None
                 for pid, s in zip(ref_ids, ref_symbols)]
        lookup = _lookup_formula(ref_rows, k_ids)
        if lookup is None and ref_rows:
            # 公式过长：只展开前 4 项，其余降为数量提示（不复制无法联查的值）。
            lookup = _lookup_formula(ref_rows[:4], k_ids[:4],
                                     f"…另有{len(ref_rows) - 4}项，完整清单见「参数引用」列")
        if lookup:
            _formula(ws, row, 11, lookup, fill=row_fill)
        else:
            put(11, "（无参数引用）" if not ref_rows else "（引用无法展开，见「参数引用」列）")

        put(12, _join_sources(op["source_refs"]))
        status = put(13, STATUS_CN[op["analysis_status"]])
        status.fill = _FILL_BLACK_CELL if blackbox else _FILL_OK
        status.font = _FONT_BLACK if blackbox else _FONT_BODY
        put(14, op["note"])
        put(15, loop_text)
        if engine is None:
            put(16, cost_text)

        explicit_ids = set(ref_ids)

        def actual_shapes(ids: list, col: int) -> None:
            expr = "=" + _actual_tensors_expr(
                ids, tensor_index, row_cells, row_guards, rows_by_symbol,
                id_of_row, explicit_ids)
            if len(expr) <= _FORMULA_LIMIT:
                _formula(ws, row, col, expr, fill=row_fill)
            else:
                put(col, "（实际 shape 公式过长，见符号 shape）")

        actual_shapes(op["inputs"], 17)
        actual_shapes(op["outputs"], 18)
        actual_shapes(op["weights"], 19)

        if engine is not None:
            rule = op["performance"]
            if rule["class"] == "container":
                _render_container_row(ws, row, rule, input_rows, inputs, layout,
                                      fill=row_fill)
                excluded_rows.append(row)
            else:
                charge = rule.get("charge") or {"mode": "cell"}
                excluded = None
                extra_note = ""
                if charge.get("mode") == "excluded":
                    charged_at = charge.get("charged_at", "")
                    if charged_at in sheet_operator_ids:
                        excluded = charge
                    else:
                        extra_note = (f"本页没有 {charged_at} 外围块，"
                                      "本行按一次计入本组小计。")
                terms = rule["terms"]
                for index, term in enumerate(terms):
                    if index > 0:
                        row += 1
                        _text(ws, row, 1, "↳ 同算子分项", fill=row_fill,
                              font=_FONT_LABEL)
                        _text(ws, row, 4, f"分项：{term['label']}", fill=row_fill)
                        if loop is not None:
                            _text(ws, row, 15, loop_text, fill=row_fill)
                    _perf_row_cells(ws, row, rule, term, perf_table, engine_refs,
                                    fill=row_fill, excluded=excluded,
                                    extra_note=extra_note)
                    if excluded is not None:
                        excluded_rows.append(row)
                    else:
                        charged_rows.append(row)
                    if len(terms) > 1:
                        _text(ws, row, PERF_FIRST_COL,
                              f"{_PERF_CLASS_CN[rule['class']]}（项 {index + 1}/{len(terms)}）",
                              fill=row_fill)
        op_rows[op["id"]] = (first_row, row)
        row += 1

    if not group["operators"]:
        _text(ws, row, 1, "（本阶段文档未提供算子）", font=_FONT_LABEL)
        row += 1
    ws.auto_filter.ref = f"A{head_row}:{get_column_letter(layout['n_cols'])}{row - 1}"

    segments: dict = {}
    for segment in group.get("segments", []):
        rows = [op_rows[oid] for oid in segment["operator_ids"] if oid in op_rows]
        if not rows:
            continue
        segments[segment["id"]] = {
            "first": min(first for first, _last in rows),
            "last": max(last for _first, last in rows),
            "count": sum(last - first + 1 for first, last in rows),
            "label": segment["label"],
        }
    return {"first_row": head_row + 1, "last_row": row - 1,
            "op_rows": op_rows, "segments": segments, "head_row": head_row,
            "charged_rows": charged_rows, "excluded_rows": excluded_rows}


def _render_network_summary(ws, start_row: int, network: dict) -> int:
    """整网外围 + 声明层组归属（只列归属与次数，不铺开层循环、不算成本）。"""
    _banner(ws, start_row,
            f"{network['label']}（外围 {len(network['periphery'])} 项、"
            f"{len(network['groups'])} 个层组、共 {network['layer_count']} 层；"
            f"成本待提供）", _FILL_OP_SECTION)
    row = start_row + 1
    _text(ws, row, 1, "网络外围路径", font=_FONT_LABEL, fill=_FILL_LABEL)
    periphery = " → ".join(
        f"{item['name']}（{item['id']} · {item['module_path']}）"
        for item in network["periphery"]) or "（未提供）"
    _text(ws, row, 2, periphery)
    row += 1
    header = ("层组id", "名称", "真实层索引", "层数", "循环次数", "成本状态")
    for col, label in enumerate(header, start=1):
        cell = _text(ws, row, col, label, font=_FONT_HEAD, fill=_FILL_OP_HEAD)
        cell.border = _BORDER
    row += 1
    total = 0
    for entry in network["groups"]:
        _text(ws, row, 1, entry["id"])
        _text(ws, row, 2, entry["label"])
        _text(ws, row, 3, ", ".join(str(index) for index in entry["layer_indices"]))
        _number(ws, row, 4, entry["count"])
        _number(ws, row, 5, entry["count"])
        _text(ws, row, 6, network["cost_status"])
        total += entry["count"]
        row += 1
    _text(ws, row, 1, "合计", font=_FONT_LABEL, fill=_FILL_LABEL)
    summary_text = (network.get("count_note") or
                    f"{len(network['groups'])} 组层数合计 {total}；"
                    f"整网声明 {network['layer_count']} 层")
    _text(ws, row, 2, summary_text)
    _text(ws, row, 6, network["cost_status"])
    row += 1
    for note in network["notes"]:
        _text(ws, row, 1, "归属说明", font=_FONT_LABEL, fill=_FILL_LABEL)
        _text(ws, row, 2, note)
        row += 1
    _text(ws, row, 1, "父子计数", font=_FONT_LABEL, fill=_FILL_LABEL)
    _text(ws, row, 2,
          "本区只列层组归属与循环次数，不写成本小计；算子区是组内明细，"
          "父组与子项不重复计入整网成本（成本实现留给用户公式提供后的后续任务）。")
    return row + 2


def _subtotal_formula(rows: list[int]) -> str:
    """成本小计：只覆盖“计入本组”的行，全部为数值才求和。

    被整网一次性收走的行（mask/RoPE 引用行）不进入区间；未定/文本行会让小计显示
    “未知（N 项未定）”，而不是被忽略空白的 SUM 悄悄跳过。
    """
    if not rows:
        return "=0"
    ref = get_column_letter(PERF_COST_COL)
    spans: list[str] = []
    start = previous = rows[0]
    for row in rows[1:]:
        if row == previous + 1:
            previous = row
            continue
        spans.append(f"{ref}{start}:{ref}{previous}" if previous != start else f"{ref}{start}")
        start = previous = row
    spans.append(f"{ref}{start}:{ref}{previous}" if previous != start else f"{ref}{start}")
    joined = ",".join(spans)
    return (f"=IF(COUNT({joined})={len(rows)},SUM({joined}),"
            f"\"未知（\"&({len(rows)}-COUNT({joined}))&\" 项未定）\")")


def _render_group_totals(ws, start_row: int, group: dict, layout: dict,
                         op_info: dict, condition_table: dict) -> dict:
    """层组/外围段的小计与层组总计行（整网汇总只引用这里，不再回到算子行）。"""
    refs: dict = {"segments": {}}
    col = get_column_letter(PERF_COST_COL)
    charged = set(op_info["charged_rows"])
    row = start_row
    for segment in group.get("segments", []):
        info = op_info["segments"].get(segment["id"])
        if info is None or not segment["subtotal"]:
            continue
        segment_rows = [index for index in range(info["first"], info["last"] + 1)
                        if index in charged]
        _text(ws, row, 1, f"【{segment['label']}】小计（{len(segment_rows)} 个计费行）",
              font=_FONT_LABEL, fill=_FILL_LABEL)
        _text(ws, row, 15, "∑ 本段计费行耗时；不乘次数")
        cell = _formula(ws, row, PERF_COST_COL, _subtotal_formula(segment_rows),
                        fill=_FILL_FORMULA)
        cell.number_format = _NUMBER_FORMAT_COST
        _text(ws, row, PERF_FIRST_COL + 7,
              "小计只在全部计费行已定时才是数值；未知行不按 0 计入；"
              "整网一次性项不在本段重复收。")
        refs["segments"][segment["id"]] = f"{col}{row}"
        row += 1
    if group["loop"] is not None:
        rows = [index for index in charged
                if op_info["first_row"] <= index <= op_info["last_row"]]
        _text(ws, row, 1, f"单层算子小计（{len(rows)} 个计费行）", font=_FONT_LABEL,
              fill=_FILL_LABEL)
        _text(ws, row, 15, "不乘层数；仅作层组总计的输入")
        cell = _formula(ws, row, PERF_COST_COL, _subtotal_formula(sorted(rows)),
                        fill=_FILL_FORMULA)
        cell.number_format = _NUMBER_FORMAT_COST
        excluded = len(op_info["excluded_rows"])
        _text(ws, row, PERF_FIRST_COL + 7,
              "同组各层同构且共享同一场景输入，单层小计×真实层数即层组总计；"
              + (f"已排除 {excluded} 行整网一次性项（mask/RoPE 引用行/容器）。"
                 if excluded else ""))
        refs["subtotal"] = f"{col}{row}"
        subtotal_ref = f"${col}${row}"
        row += 1
        count = group["loop"]["count"]
        _text(ws, row, 1, f"层组总计（单层小计 × {count} 层）", font=_FONT_LABEL,
              fill=_FILL_LABEL)
        _text(ws, row, 15, f"循环归属 {group['id']}：真实层 {len(group['layer_indices'])} 层")
        body = (f"IF(ISNUMBER({subtotal_ref}),{subtotal_ref}*{count},"
                f"\"未知（单层小计未定）\")")
        condition = group.get("execution_condition")
        if condition:
            compiled = _compile_condition(condition, condition_table,
                                          "execution_condition")
            cond_reason = _perf_unknown_text(compiled["names"])
            total_formula = (f"=IF({compiled['guard']},"
                             f"IF({compiled['expr']},{body},0),"
                             f"{_xl_quote(cond_reason)})")
        else:
            total_formula = f"={body}"
        cell = _formula(ws, row, PERF_COST_COL, total_formula, fill=_FILL_FORMULA)
        cell.number_format = _NUMBER_FORMAT_COST
        _text(ws, row, PERF_FIRST_COL + 7,
              "本行=单层小计×层数，仅作阅读核对；整网串行汇总引用单层小计并在"
              "“次数”列乘层数，两者同值且只乘一次。"
              + (f" 执行条件：{condition['when_text']}；条件不成立时本行记 0。"
                 if condition else ""))
        refs["total"] = f"{col}{row}"
        row += 1
    return refs


def _compile_condition(condition: dict, table: dict, label: str) -> dict:
    """条件（when + 可选 guards）：返回表达式、名字守卫与合并守卫。"""
    compiled = _perf_compile(condition["when"], table, label)
    extra = [_perf_compile(expr, table, f"{label}.guards") for expr in
             condition.get("guards", [])]
    guard = _and_guard([compiled["guard"]] + [
        f"IFERROR({item['expr']},FALSE)" for item in extra])
    names = sorted({*compiled["names"], *[n for item in extra for n in item["names"]]})
    return {"expr": compiled["expr"], "guard": guard, "names": names}


def _condition_table(input_rows: dict, inputs: dict) -> dict:
    """整网级条件（如 Kref>0）只能引用本表可填写输入，不借用某组参数。"""
    table = {}
    for iid, item in inputs.items():
        ref = f"$D${input_rows[iid]}"
        guard = _input_guard(item, ref)
        for key in (item["symbol"], item["id"]):
            table.setdefault(key, (ref, guard))
    return table


def _render_serial_summary(ws, start_row: int, sheet: dict, layout: dict,
                           block_refs: dict, input_rows: dict, inputs: dict) -> int:
    """整网串行求和：入口外围 → 四层组（单层小计×真实层数）→ 出口外围。"""
    network = sheet["network"]
    groups_by_id = {group["id"]: group for group in sheet["groups"]}
    network_groups = {group["id"]: group for group in network["groups"]}
    condition_table = _condition_table(input_rows, inputs)
    contributions = []
    for entry in network.get("serial_order") or []:
        group = groups_by_id.get(entry["group"])
        if group is None:
            raise WorkbookSpecError(f"串行顺序引用的组 {entry['group']!r} 不存在")
        refs = block_refs.get(group["id"], {})
        if entry["kind"] == "group":
            # 统一口径：F 列引用**单层**小计，G 列填真实层数，H=F*G 只乘一次。
            # 层组总计行（单层小计×层数）只作与 H 同值的阅读行，不被整网再引用。
            cell = refs.get("subtotal")
            if cell is None:
                raise WorkbookSpecError(
                    f"层组 {group['id']} 缺单层小计单元格（性能区未正确渲染）")
            contributions.append({
                "kind": "group", "id": group["id"], "name": group["label"],
                "layers": ", ".join(str(index) for index in group["layer_indices"]),
                "count": group["loop"]["count"], "cell": cell,
                "condition": entry.get("condition"),
            })
        else:
            info = refs.get("segments", {}).get(entry["segment"])
            if info is None:
                raise WorkbookSpecError(
                    f"段 {group['id']}/{entry['segment']} 缺小计单元格（性能区未正确渲染）")
            segment = next(segment for segment in group["segments"]
                           if segment["id"] == entry["segment"])
            contributions.append({
                "kind": "segment", "id": f"{group['id']}/{entry['segment']}",
                "name": segment["label"], "layers": "—", "count": 1, "cell": info,
                "condition": entry.get("condition"),
            })

    _banner(ws, start_row,
            f"{network['label']}（{len(network['groups'])} 个层组、共 "
            f"{network['layer_count']} 层；按串行求和，不计并行/融合/通信）",
            _FILL_OP_SECTION, n_cols=layout["n_cols"])
    row = start_row + 1
    total_layers = sum(group["count"] for group in network["groups"])
    summary_text = (network.get("count_note") or
                    f"{len(network['groups'])} 组层数合计 {total_layers}；"
                    f"整网声明 {network['layer_count']} 层")
    _text(ws, row, 1, "层组归属校验", font=_FONT_LABEL, fill=_FILL_LABEL)
    _text(ws, row, 2, summary_text)
    ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=9)
    row += 1
    header = ("序号", "段/层组", "名称", "真实层索引", "层数",
              "单次成本 ms（层组=单层小计；外围段=本段小计）", "次数", "贡献 ms",
              "成本状态")
    for index, label in enumerate(header, start=1):
        cell = _text(ws, row, index, label, font=_FONT_HEAD, fill=_FILL_OP_HEAD)
        cell.border = _BORDER
    ws.row_dimensions[row].height = 30
    row += 1
    first_contribution = row
    for index, item in enumerate(contributions, start=1):
        cell = _text(ws, row, 1, index)
        cell.alignment = _CENTER
        _text(ws, row, 2, item["id"])
        name = item["name"]
        if item["condition"]:
            name += f"（条件执行：{item['condition']['when_text']}）"
        _text(ws, row, 3, name)
        _text(ws, row, 4, item["layers"])
        _number(ws, row, 5, item["count"])
        ref = f"{get_column_letter(PERF_COST_COL + 8)}{row}"  # 贡献列
        cost = _formula(ws, row, 6, f"={item['cell']}", fill=_FILL_FORMULA)
        cost.number_format = _NUMBER_FORMAT_COST
        _number(ws, row, 7, item["count"])
        base = (f"IF(ISNUMBER($F{row}),$F{row}*$G{row},"
                f"\"未知（单次成本未定）\")")
        if item["condition"]:
            compiled = _compile_condition(item["condition"], condition_table,
                                          "serial_order.condition")
            cond_reason = _perf_unknown_text(compiled["names"])
            contrib = _formula(
                ws, row, 8,
                f"=IF({compiled['guard']},IF({compiled['expr']},{base},0),"
                f"{_xl_quote(cond_reason)})", fill=_FILL_FORMULA)
            _formula(ws, row, 9,
                     f"=IF({compiled['guard']},IF({compiled['expr']},\"已计\","
                     f"\"条件不执行（计 0）：{item['condition']['when_text']}\"),"
                     f"{_xl_quote(cond_reason)})")
        else:
            contrib = _formula(ws, row, 8, f"={base}", fill=_FILL_FORMULA)
            _formula(ws, row, 9, f"=IF(ISNUMBER($H{row}),\"已计\",\"未知\")")
        contrib.number_format = _NUMBER_FORMAT_COST
        row += 1
    last_contribution = row - 1
    span = f"H{first_contribution}:H{last_contribution}"
    count = len(contributions)

    def total_row(label: str, formula: str, note: str, fmt: str | None = None) -> None:
        nonlocal row
        cell = _text(ws, row, 2, label, font=_FONT_LABEL, fill=_FILL_LABEL)
        cell.alignment = _WRAP_TOP
        value = _formula(ws, row, 8, formula, fill=_FILL_FORMULA)
        if fmt:
            value.number_format = fmt
        _text(ws, row, 3, note)
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=4)
        ws.merge_cells(start_row=row, start_column=5, end_row=row, end_column=7)
        row += 1

    total_row("已知部分小计（仅已计项；不是整网总计）", f"=SUM({span})",
              "SUM 只累加数值单元格，未定项不按 0 计入", _NUMBER_FORMAT_COST)
    total_row("未定贡献项数", f"={count}-COUNT({span})",
              f"共 {count} 个串行贡献项（入口/出口外围段 + 各真实层组，按本表实际组数生成）")
    total_row("整网单次调用总耗时 ms",
              f"=IF(COUNT({span})={count},SUM({span}),"
              f"\"未知（\"&({count}-COUNT({span}))&\" 项贡献未定）\")",
              "只有全部贡献项已定时才是数值；否则给出原因与缺口数", _NUMBER_FORMAT_COST)
    def note_row(label: str, text: str) -> None:
        nonlocal row
        _text(ws, row, 1, label, font=_FONT_LABEL, fill=_FILL_LABEL,
              alignment=_TOP)
        _text(ws, row, 2, text)
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=9)
        ws.row_dimensions[row].height = 15 * max(1, math.ceil(len(text) / 130))
        row += 1

    for note in network["notes"]:
        note_row("归属说明", note)
    note_row(
        "父子计数",
        "按“单层小计 × 真实层数”只乘一次：F=单层小计（外围段为段小计，次数 1），"
        "G=次数，H=F×G；层组总计行（单层小计×层数）与 H 同值，仅作阅读核对，"
        "不被整网再引用。算子行是单层小计的下级明细，不重复计入。"
        "同组各层同构且共享同一场景输入，故单层成本可乘层数；"
        "MoE 专家模板按总路由分配数计，不按命中专家数机械乘代表层成本。")
    return row + 2


def _build_sheet(ws, spec: dict, sheet: dict) -> None:
    layout = _layout(spec["performance"].get("engine"))
    for idx, width in enumerate(layout["widths"], start=1):
        ws.column_dimensions[get_column_letter(idx)].width = width
    next_row = _render_sheet_summary(ws, spec, sheet, layout)
    next_row, input_rows, inputs = _render_inputs(ws, next_row, sheet, layout)
    sheet_operator_ids = {op["id"] for group in sheet["groups"]
                          for op in group["operators"]}
    block_refs: dict = {}
    for group in sheet["groups"]:
        next_row = _render_group_summary(
            ws, next_row, group, layout, spec["performance"].get("engine"))
        (next_row, row_of_id, rows_by_symbol, id_of_row, symbol_cells,
         symbol_guards, param_cells, param_guards) = _render_parameters(
            ws, next_row, group, sheet["name"], input_rows, inputs, layout)
        group = {**group, "sheet_name": sheet["name"]}
        op_info = _render_operators(
            ws, next_row, group, spec, layout, input_rows, inputs,
            row_of_id, rows_by_symbol, id_of_row, param_cells, param_guards,
            sheet_operator_ids)
        next_row = ws.max_row + 2
        if spec["performance"].get("engine") is not None:
            block_refs[group["id"]] = _render_group_totals(
                ws, next_row, group, layout, op_info,
                _condition_table(input_rows, inputs))
            next_row = ws.max_row + 2
    if sheet["network"] is not None:
        if spec["performance"].get("engine") is not None:
            _render_serial_summary(ws, next_row, sheet, layout, block_refs,
                                   input_rows, inputs)
        else:
            _render_network_summary(ws, next_row, sheet["network"])
    # 启用性能区时冻结到 C2（A=黑盒标记、B=算子id 常驻）：横向滚到性能列 T:AA 时
    # 仍能识别算子行；未启用性能区的旧路径保持原有 B2，不改旧产物。
    ws.freeze_panes = "C2" if spec["performance"].get("engine") is not None else "B2"
    ws.sheet_properties.tabColor = "FF1F4E79" if sheet["id"] == "prefill" else "FF215C36"


# ---------------------------------------------------------------- ZIP 稳定化


def _stabilize_core_props(payload: bytes) -> bytes:
    """把 core.xml 的 created/modified 钉为固定时刻（不静默放过开放形式）。"""
    text = payload.decode("utf-8")
    patched, count = _CORE_TIME_RE.subn(lambda m: m.group(1) + _FIXED_ISO + m.group(4), text)
    if count != 2:
        raise WorkbookExportError(f"无法固定工作簿时间属性（匹配 {count} 处，预期 2 处）")
    return patched.encode("utf-8")


def _normalize_zip(payload: bytes) -> bytes:
    """固定条目顺序、时间戳、权限、创建系统与属性时间，保证相同输入字节一致。"""
    source = zipfile.ZipFile(io.BytesIO(payload), "r")
    try:
        names = sorted(source.namelist(), key=lambda n: (n != "[Content_Types].xml", n))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
            for name in names:
                data = source.read(name)
                if name == "docProps/core.xml":
                    data = _stabilize_core_props(data)
                info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_DATE)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 0
                info.external_attr = 0o600 << 16
                info.internal_attr = 0
                info.comment = b""
                target.writestr(info, data)
        return buffer.getvalue()
    finally:
        source.close()


def export_workbook(workbook_spec: dict) -> bytes:
    """按通用 spec 导出 xlsx 字节（唯一创建工作簿/保存/稳定 ZIP 的实现）。

    spec 由模型适配层声明 sheet 与分组；本函数不固定 sheet 数量，也不读取模型源码。
    只读输入、只返回字节；输入不合法时抛 WorkbookExportError（ValueError 子类）。
    """
    spec = _validate_workbook_spec(workbook_spec)

    wb = Workbook()
    wb.iso_dates = True
    model = spec["model"]
    props = wb.properties
    props.creator = _FIXED_CREATOR
    props.lastModifiedBy = _FIXED_CREATOR
    props.title = f"{model['display_name']} 结构分析工作簿"
    props.description = ("schema operator_workbook.v2；工作表与分组由模型适配层声明；"
                         "由 structure_reader.export.operator_workbook 导出")
    props.subject = f"{model['model_key']} · " + " / ".join(
        sheet["name"] for sheet in spec["sheets"])
    props.created = _FIXED_EPOCH
    props.modified = _FIXED_EPOCH

    first = True
    for sheet in spec["sheets"]:
        ws = wb.active if first else wb.create_sheet()
        first = False
        ws.title = sheet["name"]
        _build_sheet(ws, spec, sheet)
    wb.active = 0

    buffer = io.BytesIO()
    wb.save(buffer)
    return _normalize_zip(buffer.getvalue())


def _legacy_reading_spec(doc: dict) -> dict:
    """把既有 hf_decoder_reading.v1 文档转成通用 spec（兼容薄适配，不另建工作簿）。"""
    scope = doc["scope"]
    model = doc["model"]
    sheets = []
    for phase in doc["phases"]:
        group = {
            "id": phase["id"],
            "label": phase["title"],
            "attention_type": scope["attention_type"],
            "moe_type": scope["moe_type"],
            "layer_indices": [scope["layer_index"]],
            "representative_layer_index": scope["layer_index"],
            "loop": {
                "count": 1,
                "label": "单层阅读表",
                "condition": "本表只分析该层一次；整网层组循环见分层入口",
                "note": "",
            },
            "template_note": "单层兼容投影：同一 Block 模板复用不代表共享权重。",
            "scope_note": (f"layer_index={scope['layer_index']}；"
                           f"attention_type={scope['attention_type']}；"
                           f"moe_type={scope['moe_type']}；"
                           f"analysis_path={scope['analysis_path']}"),
            "parameters": phase["parameters"],
            "tensors": phase["tensors"],
            "operators": phase["operators"],
            "cache_notes": phase["cache_notes"],
            "coverage_notes": phase["coverage_notes"],
        }
        sheets.append({
            "id": phase["id"],
            "name": phase["sheet_name"],
            "title": f"{phase['title']}（decoder 第 {scope['layer_index']} 层阅读表）",
            "notes": scope["notes"],
            "inputs": [],
            "groups": [group],
            "network": None,
        })
    return {
        "schema": WORKBOOK_SCHEMA,
        "model": model,
        "performance": {
            "status": "待提供（v1 兼容投影不带性能公式）",
            "note": "用户性能公式与 XPU 能力尚未提供；不填猜测耗时、不把缺失成本记 0",
        },
        "disclaimer": DISCLAIMER,
        "sheets": sheets,
    }


def export_operator_workbook(reading_document: dict) -> bytes:
    """v1 兼容薄适配：校验 hf_decoder_reading.v1 后交给同一个通用导出器。

    保留该入口只为既有冻结接口/测试与外部调用；真实单层/分组生产路径已在
    ``build_hf_decoder`` 改用通用 spec。它不创建第二套 Workbook，也不私有复制
    导出流程。
    """
    doc = _validate_document(reading_document)
    return export_workbook(_legacy_reading_spec(doc))
