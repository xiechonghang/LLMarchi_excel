"""HF decoder 阅读样板构建编排（里程碑一，独立于旧 build_pipeline 主干）。

职责：读取注册 → 有效配置 → reading_document → 通用/语义检查 → HTML/xlsx
生成 → 内容寻址落盘（.work 暂存 + os.replace 原子提交 + 一致性核对 +
_failed 诊断），复用 cli._materialize 的落盘与失败隔离机制。

与旧 run_build 的关系：旧 run_build（build_pipeline.py:326）绑定参考实现
DSpark 链，不进本路径；本模块只服务新注册键 deepseek_v4_flash_0731_hf。
analysis_id 覆盖：根配置/源码/模板/导出器/渲染器/规则/提取器源码 SHA256 +
layer_index + 场景标识，不含时间戳/绝对路径。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from structure_reader.extract.deepseek_v4_hf import (
    ReadingDocError,
    build_reading_document,
)
from structure_reader.render.decoder_page import DecoderPageError, render_decoder_page
from structure_reader.rules.deepseek_v4_hf import HfConfigError, SOURCE_FILES

__all__ = ["HfBuildError", "HfBuildResult", "run_hf_build"]


class HfBuildError(ValueError):
    """HF build 失败（具名）：输入/提取/检查/渲染/导出/落盘任何一步不通过。"""


@dataclass(frozen=True)
class HfBuildResult:
    """一次 HF build 的产物位置与摘要。"""

    output_dir: Path
    analysis_id: str
    reused: bool
    model_key: str
    layer_index: int


# analysis_id 覆盖的源码（相对 structure_reader 工程根）：
# 真正影响 HF 产物的全部依赖；旧 DSpark/参考实现源码不进入（源码变化不应
# 使 HF 候选伪变更）。
_TOOL_SOURCE_RELPATHS = (
    "src/structure_reader/rules/deepseek_v4_hf.py",
    "src/structure_reader/extract/deepseek_v4_hf.py",
    "src/structure_reader/checks/deepseek_v4_hf_contracts.py",
    "src/structure_reader/render/decoder_page.py",
    "src/structure_reader/render/diagram.py",
    "src/structure_reader/export/operator_workbook.py",
    "src/structure_reader/build_hf_decoder.py",
    "src/structure_reader/model_registry.py",
    "src/structure_reader/graph_document.py",
    "src/structure_reader/graph_ir.py",
    "src/structure_reader/scenario_evaluator.py",
    "configs/models/deepseek_v4_flash_0731_hf.json",
)

_DELIVERABLES = (
    "README.md",
    "run_manifest.json",
    "data/reading_document.json",
    "data/effective_config.json",
    "reports/source_manifest.json",
    "viewer/decoder.html",
    "tables/decoder_reading.xlsx",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _pretty_bytes(obj: object) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


# 本提取器分析结论只对以下注册键有效：build-hf 的模型事实（行区间、指纹、
# 结构解释）是 DeepSeek-V4 专用适配，拒绝借用给未知/其他模型键。
SUPPORTED_MODEL_KEYS = ("deepseek_v4_flash_0731_hf",)


def run_hf_build(
    workspace_root: Path,
    project_root: Path,
    layer_index: int = 4,
    model_key: str = "deepseek_v4_flash_0731_hf",
) -> tuple[HfBuildResult, dict[str, bytes]]:
    """HF 阅读样板构建：生成全部交付文件字节（不落盘，由 CLI 层 _materialize 提交）。

    返回 (result, documents)；documents 键含子目录相对路径。
    xlsx 依赖缺失（ImportError）时如实抛 HfBuildError，不降级。
    """
    workspace_root = Path(workspace_root)
    project_root = Path(project_root)
    if model_key not in SUPPORTED_MODEL_KEYS:
        raise HfBuildError(
            f"未知模型键 {model_key!r}：HF 阅读样板只支持 "
            f"{SUPPORTED_MODEL_KEYS}（模型事实为 DeepSeek-V4 专用适配，不借用）"
        )

    # 0) 来源身份：已分析指纹复算（漂移即拒绝，要求重新分析）
    from structure_reader.rules.deepseek_v4_hf import verify_source_fingerprints

    verify_source_fingerprints(workspace_root)

    # 1) 输入身份：SOURCE_FILES 实际 SHA256（失配/缺失如实报错）
    input_shas = {}
    for key, relpath in SOURCE_FILES.items():
        path = workspace_root / relpath
        if not path.is_file():
            raise HfBuildError(f"HF 输入缺失: {relpath}（不下载、不伪造）")
        input_shas[key] = _sha256_file(path)

    # 2) reading_document
    try:
        document = build_reading_document(workspace_root, layer_index=layer_index, model_key=model_key)
    except (ReadingDocError, HfConfigError) as exc:
        raise HfBuildError(f"阅读数据构建失败: {exc}") from exc

    # 3) 语义检查（新增 HF 契约；失败即 HfBuildError）
    from structure_reader.checks.deepseek_v4_hf_contracts import validate_hf_reading_document

    issues = validate_hf_reading_document(document, input_shas)
    if issues:
        raise HfBuildError(
            "HF 阅读数据检查未通过: " + "; ".join(issues[:8])
            + (f"（共 {len(issues)} 条）" if len(issues) > 8 else "")
        )

    # 4) HTML 与 xlsx（同源：同一 document）
    try:
        html_bytes = render_decoder_page(document)
    except DecoderPageError as exc:
        raise HfBuildError(f"HTML 渲染失败: {exc}") from exc
    try:
        from structure_reader.export.operator_workbook import export_operator_workbook

        xlsx_bytes = export_operator_workbook(document)
    except ImportError as exc:
        raise HfBuildError(
            f"xlsx 导出必需依赖缺失（openpyxl）: {exc}；不降级为 CSV，不发布假完整候选"
        ) from exc
    except ValueError as exc:
        raise HfBuildError(f"xlsx 导出失败（导出器拒绝输入）: {exc}") from exc

    # 5) analysis_id：输入 SHA + 工具源码 SHA + layer_index
    tool_sources = {}
    for relpath in _TOOL_SOURCE_RELPATHS:
        path = project_root / relpath
        tool_sources[relpath] = _sha256_file(path) if path.is_file() else "MISSING"
    identity = {
        "schema": "structure_reader_hf_build_identity_v1",
        "model_key": model_key,
        "layer_index": layer_index,
        "analysis_path": "eager",
        "input_sha256": input_shas,
        "tool_sources": tool_sources,
    }
    analysis_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()

    # 6) 交付文档
    from structure_reader.rules.deepseek_v4_hf import (
        effective_parameter_records,
        load_hf_model_values,
    )

    values = load_hf_model_values(workspace_root, model_key=model_key)
    records = effective_parameter_records(values)
    documents = {
        "README.md": _readme(document, analysis_id),
        "data/reading_document.json": _pretty_bytes(document),
        "data/effective_config.json": _pretty_bytes({
            "schema": "hf_effective_config_v1",
            "model_key": model_key,
            "source_revision": values.source_revision,
            "values": {f: getattr(values, f) for f in (
                "hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "vocab_size", "rms_norm_eps",
                "hidden_act", "use_cache", "sliding_window", "max_position_embeddings",
                "q_lora_rank", "o_groups", "o_lora_rank", "qk_rope_head_dim",
                "partial_rotary_factor", "rope_main_theta", "rope_compress_theta",
                "rope_compress_factor", "compress_rate_csa", "compress_rate_hca",
                "index_n_heads", "index_head_dim", "index_topk", "hc_mult",
                "hc_sinkhorn_iters", "hc_eps", "n_routed_experts", "n_shared_experts",
                "num_experts_per_tok", "scoring_func", "norm_topk_prob",
                "routed_scaling_factor", "moe_intermediate_size", "swiglu_limit",
            )},
            "layer_types": values.layer_types,
            "mlp_layer_types": values.mlp_layer_types,
            "parameter_records": records,
        }),
        "reports/source_manifest.json": _pretty_bytes({
            "schema": "hf_source_manifest_v1",
            "model_key": model_key,
            "source_revision": values.source_revision,
            "inputs": [
                {"key": key, "path": relpath, "sha256": input_shas[key]}
                for key, relpath in SOURCE_FILES.items()
            ],
            "tool_sources": tool_sources,
            "identity": identity,
            "analysis_id": analysis_id,
        }),
        "viewer/decoder.html": html_bytes,
        "tables/decoder_reading.xlsx": xlsx_bytes,
    }
    doc_hashes = {name: _sha256_bytes(data) for name, data in documents.items()}
    deliverables = {name: doc_hashes[name] for name in _DELIVERABLES if name in doc_hashes}
    documents["run_manifest.json"] = _pretty_bytes({
        "schema": "hf_decoder_build_manifest_v1",
        "status": "success",
        "model_key": model_key,
        "analysis_id": analysis_id,
        "layer_index": layer_index,
        "deliverables": deliverables,
        "identity": identity,
    })
    return (
        HfBuildResult(
            output_dir=project_root / "outputs" / model_key / analysis_id,
            analysis_id=analysis_id,
            reused=False,
            model_key=model_key,
            layer_index=layer_index,
        ),
        documents,
    )


def _readme(document: dict, analysis_id: str) -> bytes:
    model = document["model"]
    scope = document["scope"]
    lines = [
        f"# {model['display_name']} — HF Decoder 阅读样板（里程碑一）",
        "",
        f"- analysis_id: `{analysis_id}`",
        f"- 模型键: `{model['model_key']}`（implementation=transformers, eager）",
        f"- 源码版本: `{model['source_revision']}`",
        f"- 配置: `{model['config_path']}`",
        f"- 样板层: layer_idx={scope['layer_index']}（{scope['attention_type']} + {scope['moe_type']}）",
        "",
        "## 交付文件",
        "",
        "- `viewer/decoder.html` — 单文件离线阅读页：先完整 Prefill，再完整 Decode，连续展示。",
        "- `tables/decoder_reading.xlsx` — 两表 Excel（Prefill/Decode），每表自带参数符号区与算子区，与 HTML 同源。",
        "- `data/reading_document.json` — hf_decoder_reading.v1 冻结接口阅读数据（HTML/Excel 同一来源）。",
        "- `data/effective_config.json` — HF 有效配置与逐参数来源记录。",
        "- `reports/source_manifest.json` — 输入/工具源码 SHA256 清单与 analysis_id 身份。",
        "",
        "## 范围声明",
        "",
    ]
    lines.extend(f"- {note}" for note in scope.get("notes", []))
    lines.extend([
        "",
        "## 使用方式",
        "",
        "浏览器直接打开 `viewer/decoder.html`（无需网络）；"
        "Excel 打开 `tables/decoder_reading.xlsx`（两表，参数引用可点击联查本表参数区）。",
        "",
        "本样板不加载权重、不运行模型；正确性目标为静态结构一致。",
    ])
    return ("\n".join(lines) + "\n").encode("utf-8")
