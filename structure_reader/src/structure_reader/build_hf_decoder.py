"""HF 阅读产物构建编排（单层入口 / 分层入口共用一条流水线，按模型适配分派）。

职责：来源核验 → 输入/工具身份 → 参数化 Block 结构（一份）+ 场景投影 →
通用/语义检查 → 源码快照 → 可选模块级 ONNX/onnxsim（仅 DeepSeek 适配实现，
GLM-5.3 明确拒绝）→ 分层浏览 HTML / 单层 HTML / xlsx 生成 → 产物装配
（source_manifest + run_manifest）→ 内容寻址落盘（.cli._materialize）。

模型适配（``_ADAPTERS``）只声明模型事实与入口，不复制编排：
- DeepSeek-V4-Flash：``rules/deepseek_v4_hf.py`` + ``extract/deepseek_v4_hf.py``
  + ``extract/deepseek_v4_network.py`` + ``export/hf_workbook.py``；
- GLM-5.3：``rules/glm53_hf.py`` + ``extract/glm53_hf.py`` +
  ``extract/glm53_network.py`` + ``export/glm53_workbook.py``。
- GLM-5.3-Flash：``rules/glm53flash_hf.py`` + ``extract/glm53flash_hf.py`` +
  ``extract/glm53flash_network.py`` + ``export/glm53flash_workbook.py``（独立适配，
  不预设与 GLM-5.3 相同的 KDA/mHC/NoPE/k-pool 语义）。
- Qwen3.8-Flash-Next：``rules/qwen38flash_hf.py`` + ``extract/qwen38flash_hf.py`` +
  ``extract/qwen38flash_network.py`` + ``export/qwen38flash_workbook.py``（独立适配，
  不预设 Qwen3-Next/Qwen3.5 的 GQA/线性注意力语义；GDN/QSA/4 路 Gated Residual/
  N-gram PLE 各自表达）。
- HiDream-O1-Image：``rules/hidream_o1_hf.py`` + ``extract/hidream_o1_hf.py`` +
  ``extract/hidream_o1_network.py`` + ``export/hidream_o1_workbook.py``（**单次**生成
  网络，不套 Decoder/Prefill/Decode：文本 token + 像素 patch + 时间步 → x_pred；
  36 层统一主干 + 条件执行的 27 个 ViT block；单 sheet 工作簿）。

``--all-variants`` 路径发布分层浏览页 ``viewer/structure.html``（整网→层组→Block→
算子，默认只显示当前层级）；两页 Excel 由同一份结构的场景投影生成。单层入口
（``viewer/decoder.html`` + ``tables/decoder_reading.xlsx``）保留给 DeepSeek；
GLM-5.3 只提供分层入口，未实现单层长页时不伪造。

analysis_id 覆盖：根配置/源码/模板/导出器/渲染器/规则/提取器源码 SHA256 +
层组集合/场景标识，不含时间戳/绝对路径。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from structure_reader.render.archify_layers import _edge_label_box_width
from structure_reader.render.decoder_page import DecoderPageError, render_decoder_page
from structure_reader.render.hf_model_profile import (
    DEEPSEEK_PAGE_PROFILE,
    GLM53_PAGE_PROFILE,
    GLM53FLASH_PAGE_PROFILE,
    HIDREAM_O1_PAGE_PROFILE,
    QWEN38FLASH_PAGE_PROFILE,
    PageProfile,
)
from structure_reader.render.hf_structure_page import (
    StructurePageError,
    render_structure_page,
)

__all__ = ["HfBuildError", "HfBuildResult", "run_hf_build", "SUPPORTED_MODEL_KEYS"]

# 分层 Archify 图：五模型共用一套构建器，模型差异在 render/archify_profiles.py 声明。
from structure_reader.render.archify_profiles import (
    MODEL_GRAPH_PROFILES,
    graph_profile_for,
)

ARCHIFY_LAYER_MODELS = tuple(MODEL_GRAPH_PROFILES)
ARCHIFY_REPOSITORY = graph_profile_for(
    "deepseek_v4_flash_0731_hf").repository.as_finalize()
ARCHIFY_LEVELS = graph_profile_for(
    "deepseek_v4_flash_0731_hf").levels
_PKG_SOURCE_DIR = Path(__file__).resolve().parent


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
    all_variants: bool = False
    variants: tuple = ()


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


# 单层入口的交付文件（历史清单，保留供审阅对照；运行时不作为发布白名单）。
_DELIVERABLES = (
    "README.md",
    "run_manifest.json",
    "data/reading_document.json",
    "data/effective_config.json",
    "reports/source_manifest.json",
    "viewer/decoder.html",
    "tables/decoder_reading.xlsx",
)



# ---------------------------------------------------------------------------
# 模型适配声明
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Adapter:
    """一个 HF 模型的声明性适配：模型事实、能力入口与少量模型特有文案。

    五个模型共用的编排（层组容器、页面条目、身份/清单结构、有效配置、导出、落盘）
    由下面的 ``_page_items`` / ``_shipped_structures`` / ``_identity_structures`` /
    ``_run_manifest_structure`` / ``_effective_config`` 按本声明生成；这里只声明
    “哪些真实字段/能力属于该模型”，不复制同一段闭包五次。
    """

    model_key: str
    sample_layer_index: int
    supports_single: bool
    supports_onnx: bool
    source_files: dict
    verify_sources: Callable
    tool_sources: tuple
    onnx_tool_sources: tuple
    load_values: Callable
    layer_documents: Callable
    structures: Callable
    network: Callable
    validate_document: Callable
    validate_layer_structure: Callable
    validate_network: Callable
    page_profile: PageProfile
    export_structures: Callable
    export_single: Callable
    readme_single: Callable
    readme_variants: Callable
    identity_schema: str
    effective_value_fields: tuple
    effective_parameter_records: Callable
    # 页面条目/身份/run_manifest 的模型特有字段（共享字段由公共函数补齐）。
    page_item_extra: tuple = ()
    identity_extra: tuple = ()
    manifest_group_fields: tuple = ()
    manifest_network_fields: tuple = ()
    manifest_fixed: tuple = ()
    # 有效配置的顶层标量/逐层字段；模型只在元组里声明自己真实拥有的键。
    effective_extra_fields: tuple = ()
    effective_layer_fields: tuple = ()
    # 分层入口是否同时写 data/effective_config.json（DeepSeek 沿用历史清单，不写）。
    variant_effective_config: bool = False
    onnx_expected: dict | None = None
    # 可选性能表导出器：只有已接入新布局的模型声明 (container, network) -> bytes。
    # None 表示该模型不产出 tables/decoder_performance.xlsx，不借用其它模型的布局。
    export_performance: Callable | None = None


# 五个模型页面条目/身份共用的真实字段；模型特有字段放在中间。
_PAGE_ITEM_BASE = ("id", "label", "attention_type", "moe_type")
_PAGE_ITEM_TAIL = ("layer_indices", "representative_layer_index", "structure")


def _page_items(adapter: _Adapter, variants: list[dict]) -> list[dict]:
    """分层页条目：共享字段 + 模型特有字段，不复制五份同形闭包。"""
    keys = _PAGE_ITEM_BASE + adapter.page_item_extra + _PAGE_ITEM_TAIL
    return [{key: variant[key] for key in keys} for variant in variants]


def _shipped_structures(adapter: _Adapter, container: dict) -> dict:
    """落盘的 data/layer_structures.json：容器 schema/model + 页面条目。"""
    return {
        "schema": container["schema"],
        "model": container["model"],
        "variants": _page_items(adapter, container["variants"]),
    }


def _identity_structures(adapter: _Adapter, variants: list[dict]) -> list[dict]:
    """analysis_id 里的层组身份：真实层索引、代表层、场景与模型特有字段。"""
    return [
        {
            "id": variant["id"],
            "layers": variant["layer_indices"],
            "representative": variant["representative_layer_index"],
            **{field: variant[field] for field in adapter.identity_extra},
            "scenarios": [s["id"] for s in variant["structure"]["scope"]["scenarios"]],
            "default_scenario": variant["structure"]["scope"]["default_scenario"],
        }
        for variant in variants
    ]


def _run_manifest_structure(adapter: _Adapter, network: dict, page_items: list[dict],
                            shipped_structures: dict) -> dict:
    """run_manifest.structure：整网/层组摘要 + 模型特有清单字段。"""
    structure = {
        "schema": shipped_structures["schema"],
        "entry": "viewer/structure.html",
        "levels": ["network", "group", "block", "operators"],
    }
    structure.update(
        {field: network["scope"][field] for field in adapter.manifest_network_fields})
    structure["layer_count"] = network["scope"]["layer_count"]
    structure["groups"] = {g["id"]: g["layer_indices"] for g in network["groups"]}
    for manifest_key, group_key in adapter.manifest_group_fields:
        structure[manifest_key] = {g["id"]: g[group_key] for g in network["groups"]}
    structure["default_scenario"] = page_items[0]["structure"]["scope"]["default_scenario"]
    structure["scenarios"] = [
        s["id"] for s in page_items[0]["structure"]["scope"]["scenarios"]]
    structure.update(dict(adapter.manifest_fixed))
    return structure


def _effective_config(adapter: _Adapter, values) -> dict:
    """data/effective_config.json：同一外层结构，模型只声明自己的字段集。"""
    document = {
        "schema": "hf_effective_config_v1",
        "model_key": values.model_key,
        "source_revision": values.source_revision,
        "values": {
            field: getattr(values, field) for field in adapter.effective_value_fields},
        **{field: getattr(values, field) for field in adapter.effective_extra_fields},
        **{field: getattr(values, field) for field in adapter.effective_layer_fields},
    }
    document["parameter_records"] = adapter.effective_parameter_records(values)
    return document


def _structure_builder(load_values: Callable, groups_fn: Callable,
                       layer_documents: Callable) -> Callable:
    """Decoder 同形层组容器构建：一份参数化 Block 结构 + 真实层归属/代表层。"""
    from structure_reader.extract.hf_layer_structures import build_structures_container

    def structures(workspace_root, model_key):
        return build_structures_container(
            workspace_root, model_key=model_key,
            values_loader=lambda root, model_key: load_values(root, model_key=model_key),
            groups_fn=groups_fn,
            build_documents_fn=lambda root, layer_index, model_key: layer_documents(
                root, layer_index=layer_index, model_key=model_key))

    return structures


# 通用工具源（两模型共用；模型专属模块另列）。
_SHARED_TOOL_SOURCES = (
    "src/structure_reader/extract/hf_scenario.py",
    "src/structure_reader/extract/hf_layer_structures.py",
    "src/structure_reader/render/decoder_page.py",
    "src/structure_reader/render/detail_render.py",
    "src/structure_reader/render/hf_structure_page.py",
    "src/structure_reader/render/hf_model_profile.py",
    "src/structure_reader/render/module_narrative.py",
    "src/structure_reader/render/display_names.py",
    "src/structure_reader/render/diagram.py",
    "src/structure_reader/render/diagram_source_labels.py",
    "src/structure_reader/export/operator_workbook.py",
    "src/structure_reader/export/hf_workbook_common.py",
    "src/structure_reader/export/perf_engine.py",
    "src/structure_reader/cli.py",
    "src/structure_reader/build_hf_decoder.py",
    "src/structure_reader/model_registry.py",
    "src/structure_reader/graph_document.py",
    "src/structure_reader/graph_ir.py",
    "src/structure_reader/scenario_evaluator.py",
    "src/structure_reader/checks/hf_contract_core.py",
)

# DeepSeek 适配的工具源：T04 既有集合 + 本票提炼出的公共模块（影响同一输出即计入
# 身份）；顺序固定，源码变化即产生新 analysis_id。
_DEEPSEEK_TOOL_SOURCES = (
    "src/structure_reader/rules/deepseek_v4_hf.py",
    "src/structure_reader/extract/deepseek_v4_hf.py",
    "src/structure_reader/extract/deepseek_v4_network.py",
    "src/structure_reader/extract/m2_hca.py",
    "src/structure_reader/extract/m2_swa_hash.py",
    "src/structure_reader/extract/hf_scenario.py",
    "src/structure_reader/extract/hf_layer_structures.py",
    "src/structure_reader/checks/deepseek_v4_hf_contracts.py",
    "src/structure_reader/checks/hf_contract_core.py",
    "src/structure_reader/render/decoder_page.py",
    "src/structure_reader/render/detail_render.py",
    "src/structure_reader/render/hf_structure_page.py",
    "src/structure_reader/render/hf_model_profile.py",
    "src/structure_reader/render/module_narrative.py",
    "src/structure_reader/render/display_names.py",
    "src/structure_reader/render/diagram.py",
    "src/structure_reader/render/diagram_source_labels.py",
    "src/structure_reader/export/operator_workbook.py",
    "src/structure_reader/export/hf_workbook_common.py",
    "src/structure_reader/export/hf_workbook.py",
    "src/structure_reader/export/deepseek_perf_rules.py",
    "src/structure_reader/export/deepseek_perf_layout.py",
    "src/structure_reader/export/deepseek_perf_layout_v2.py",
    "src/structure_reader/export/perf_engine.py",
    "src/structure_reader/cli.py",
    "src/structure_reader/build_hf_decoder.py",
    "src/structure_reader/model_registry.py",
    "src/structure_reader/graph_document.py",
    "src/structure_reader/graph_ir.py",
    "src/structure_reader/scenario_evaluator.py",
    "configs/models/deepseek_v4_flash_0731_hf.json",
)

_GLM53_TOOL_SOURCES = _SHARED_TOOL_SOURCES + (
    "src/structure_reader/checks/glm53_hf_contracts.py",
    "src/structure_reader/rules/glm53_hf.py",
    "src/structure_reader/extract/glm53_hf.py",
    "src/structure_reader/extract/glm53_network.py",
    "src/structure_reader/render/glm53_narrative.py",
    "src/structure_reader/render/glm53_source_labels.py",
    "src/structure_reader/export/glm53_workbook.py",
    "src/structure_reader/export/glm53_perf_rules.py",
)

_GLM53FLASH_TOOL_SOURCES = _SHARED_TOOL_SOURCES + (
    "src/structure_reader/checks/glm53flash_hf_contracts.py",
    "src/structure_reader/rules/glm53flash_hf.py",
    "src/structure_reader/extract/glm53flash_hf.py",
    "src/structure_reader/extract/glm53flash_network.py",
    "src/structure_reader/render/glm53flash_narrative.py",
    "src/structure_reader/render/glm53flash_source_labels.py",
    "src/structure_reader/export/glm53flash_workbook.py",
    "src/structure_reader/export/glm53flash_perf_rules.py",
)

_QWEN38FLASH_TOOL_SOURCES = _SHARED_TOOL_SOURCES + (
    "src/structure_reader/checks/qwen38flash_hf_contracts.py",
    "src/structure_reader/rules/qwen38flash_hf.py",
    "src/structure_reader/extract/qwen38flash_hf.py",
    "src/structure_reader/extract/qwen38flash_network.py",
    "src/structure_reader/render/qwen38flash_narrative.py",
    "src/structure_reader/render/qwen38flash_source_labels.py",
    "src/structure_reader/export/qwen38flash_workbook.py",
    "src/structure_reader/export/qwen38flash_perf_rules.py",
)

_HIDREAM_O1_TOOL_SOURCES = _SHARED_TOOL_SOURCES + (
    "src/structure_reader/checks/hidream_o1_hf_contracts.py",
    "src/structure_reader/rules/hidream_o1_hf.py",
    "src/structure_reader/extract/hidream_o1_hf.py",
    "src/structure_reader/extract/hidream_o1_network.py",
    "src/structure_reader/render/hidream_o1_narrative.py",
    "src/structure_reader/render/hidream_o1_source_labels.py",
    "src/structure_reader/export/hidream_o1_workbook.py",
    "src/structure_reader/export/hidream_o1_perf_rules.py",
)

_DEEPSEEK_ONNX_TOOL_SOURCES = (
    "src/structure_reader/onnx_module/__init__.py",
    "src/structure_reader/onnx_module/modeling.py",
    "src/structure_reader/onnx_module/builders.py",
    "src/structure_reader/onnx_module/builders_m2.py",
    "src/structure_reader/onnx_module/build.py",
    "src/structure_reader/onnx_module/view.py",
    "tools/netron_launcher/main.swift",
    "tools/netron_launcher/Info.plist",
    "tools/netron_launcher/setup.sh",
)

# ---------------------------------------------------------------------------
# DeepSeek 适配
# ---------------------------------------------------------------------------


def _deepseek_adapter() -> _Adapter:
    from structure_reader.checks.deepseek_v4_hf_contracts import (
        validate_hf_layer_structure,
        validate_hf_network_overview,
        validate_hf_reading_document,
    )
    from structure_reader.extract.deepseek_v4_hf import build_layer_documents
    from structure_reader.extract.deepseek_v4_network import build_network_overview
    from structure_reader.export.hf_workbook import (
        export_single_structure_workbook,
        export_structures_workbook,
    )
    from structure_reader.rules.deepseek_v4_hf import (
        SOURCE_FILES,
        effective_parameter_records,
        layer_group_assignments,
        load_hf_model_values,
        verify_source_fingerprints,
    )

    def readme_variants(analysis_id, shipped_structures, network, onnx_modules):
        return (
            "# DeepSeek-V4-Flash 分层结构浏览（整网 → 层组 → Block → 算子）\n\n"
            f"- analysis_id: `{analysis_id}`\n"
            "- 覆盖：SWA+Hash(0,1)、CSA+Hash(2)、HCA+TopK(3,5..41)、CSA+TopK(4,6..42)\n"
            "- 入口：`viewer/structure.html`（默认只显示当前层级；整网→层组→Block→内部算子，"
            "面包屑与链接可返回上级；场景参数 seqlen/kvcachelen 在同一结构上切换）\n"
            "- 数据：`data/layer_structures.json`（每层组一份参数化 Block 结构 + 场景派生）、"
            "`data/network_overview.json`（token→logits 整网路径）\n"
            "- 性能表：`tables/decoder_performance.xlsx`（已确认的共用模板布局；Prefill/Decode "
            "两页 + 模板映射/参数/项分解/原始参数辅助表；共用 mHC、Q/KV 前置与 O/残差按同构"
            "模板只展示一次并乘真实使用次数，参数层校验非法维度/吞吐为未知，活公式驱动成本）\n"
            "- 兼容表：`tables/decoder_variants.xlsx`（旧阅读接口，由旧导出器生成、字节保持；"
            "Prefill/Decode 两页；四类层组 + 整网外围；seqlen/kvcachelen/B/精度/位宽/CUBE 与 "
            "VEC 吞吐为可填写示例配置，派生值由 Excel 公式重算；逐算子 CUBE/VEC 分类、"
            "B/M/K/N 或 M/N、工作量与用户规则耗时 ms，层组小计与整网串行汇总；未知/非法输入"
            "显示未知，不按 0 计；与性能表并存但内容身份不同）\n"
            "- 构建命令：`python -c \"import sys; from structure_reader.cli import main; "
            "sys.exit(main(['build-hf','--model','deepseek_v4_flash_0731_hf',"
            "'--all-variants','--archify-layers','--onnx','--json']))\"`（项目内隔离 ONNX 依赖）\n"
            + ("- 四组 ONNX/onnxsim 与本地 Netron 链接已接入 `onnx/<group>/<phase>/<module>/`，"
               "逐组报告 `onnx/<group>/reports/onnx_build.json`。\n"
               if onnx_modules else "- 本段不含 ONNX/Netron。\n")
            + "- 限制：FP32 阅读特化、无真实权重、非 HF 数值等价；专家模板单份；"
              "mask 为默认因果/滑窗，padding 路径未覆盖。\n").encode("utf-8")

    def export_performance(container, network=None):
        """正式性能表：layout-v2-rework 适配器吃同一份真实 structures/network。

        惰性导入：只有真正生成 ``tables/decoder_performance.xlsx`` 时才加载该
        布局模块链（含 openpyxl）；其他模型与其他路径不触发 DeepSeek 专属依赖，
        缺依赖时由 ``_export_performance_workbook`` 统一转成具名 HfBuildError。
        """
        from structure_reader.export.deepseek_perf_layout_v2 import (
            export_perf_layout_v2_workbook,
        )

        return export_perf_layout_v2_workbook(container, network)

    return _Adapter(
        model_key="deepseek_v4_flash_0731_hf",
        sample_layer_index=4,
        supports_single=True,
        supports_onnx=True,
        source_files=SOURCE_FILES,
        verify_sources=verify_source_fingerprints,
        tool_sources=_DEEPSEEK_TOOL_SOURCES,
        onnx_tool_sources=_DEEPSEEK_ONNX_TOOL_SOURCES,
        load_values=load_hf_model_values,
        layer_documents=build_layer_documents,
        structures=_structure_builder(
            load_hf_model_values, layer_group_assignments, build_layer_documents),
        network=build_network_overview,
        validate_document=validate_hf_reading_document,
        validate_layer_structure=validate_hf_layer_structure,
        validate_network=validate_hf_network_overview,
        page_profile=DEEPSEEK_PAGE_PROFILE,
        export_structures=export_structures_workbook,
        export_single=export_single_structure_workbook,
        readme_single=_deepseek_readme_single,
        readme_variants=readme_variants,
        identity_schema="hf_layer_structures_identity_v1",
        effective_value_fields=(
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
        ),
        effective_layer_fields=("layer_types", "mlp_layer_types"),
        effective_parameter_records=effective_parameter_records,
        onnx_expected={"swa-hash": 16, "csa-hash": 18, "hca-topk": 17, "csa-topk": 18},
        export_performance=export_performance,
    )


def _deepseek_readme_single(document: dict, analysis_id: str) -> bytes:
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
        "- `tables/decoder_reading.xlsx` — 两页 Excel（Prefill/Decode）；每页有可填写的 "
        "seqlen/kvcachelen/B/精度/位宽示例配置，派生值由 Excel 公式重算，与 HTML 同源。",
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


# ---------------------------------------------------------------------------
# GLM-5.3 适配
# ---------------------------------------------------------------------------


def _glm53_adapter() -> _Adapter:
    from structure_reader.checks.glm53_hf_contracts import (
        validate_hf_layer_structure,
        validate_hf_network_overview,
        validate_hf_reading_document,
    )
    from structure_reader.extract.glm53_hf import build_layer_documents
    from structure_reader.extract.glm53_network import build_network_overview
    from structure_reader.export.glm53_workbook import (
        export_single_structure_workbook,
        export_structures_workbook,
    )
    from structure_reader.rules.glm53_hf import (
        SOURCE_FILES,
        effective_parameter_records,
        layer_group_assignments,
        load_hf_model_values,
        verify_source_fingerprints,
    )

    def readme_variants(analysis_id, shipped_structures, network, onnx_modules):
        return (
            "# GLM-5.3 分层结构浏览（整网 → 层组 → Block → 算子）\n\n"
            f"- analysis_id: `{analysis_id}`\n"
            "- 官方来源：`zai-org/GLM-5.3` revision "
            f"`{network['model'].get('model_revision')}`，model_type=`glm_moe_dsa`；"
            "实现为本地官方 transformers 检出（见 `reports/source_manifest.json`）。\n"
            "- 覆盖：语言主干 token→logits（Embedding → 78 层 → Final RMSNorm → "
            "logits slice → LM head）。三组真实层型：Full DSA Indexer + Dense MLP(0,1,2)、"
            "Shared DSA Indexer + TopK MoE（57 层）、Full DSA Indexer + TopK MoE（18 层）。\n"
            "- 入口：`viewer/structure.html`（默认只显示当前层级；整网→层组→Block→内部算子，"
            "面包屑与链接可返回上级；场景参数 seqlen/kvcachelen 在同一结构上切换）\n"
            "- 数据：`data/layer_structures.json`（每组一份参数化 Block 结构 + 场景派生）、"
            "`data/network_overview.json`（token→logits 整网路径）、"
            "`data/effective_config.json`（有效配置与逐参数来源）。\n"
            "- 表：`tables/decoder_variants.xlsx`（Prefill/Decode 两页；每页三组，"
            "seqlen/kvcachelen/B/精度/位宽/CUBE 与 VEC 吞吐为可填写示例配置，派生值由 Excel "
            "公式重算；逐算子 CUBE/VEC 分类、B/M/K/N 或 M/N、工作量与用户规则耗时 ms，"
            "层组小计与整网串行汇总；未知/非法输入显示未知，不按 0 计）。\n"
            "- 构建命令：`PYTHONPATH=structure_reader/src python "
            "structure_reader/scripts/read_model_structure.py build-hf --model glm_5_3_hf "
            "--all-variants --archify-layers --json`\n"
            "- 限制：只做语言主干；视觉/图像输入路径不在本票范围；MTP 层 78 不在主干；"
            "未实现可选 ONNX（显式拒绝，不静默忽略）；FP8 量化执行不分析；"
            "无真实权重、非 HF 数值等价。\n").encode("utf-8")

    return _Adapter(
        model_key="glm_5_3_hf",
        sample_layer_index=0,
        supports_single=False,
        supports_onnx=False,
        source_files=SOURCE_FILES,
        verify_sources=verify_source_fingerprints,
        tool_sources=_GLM53_TOOL_SOURCES,
        onnx_tool_sources=(),
        load_values=load_hf_model_values,
        layer_documents=build_layer_documents,
        structures=_structure_builder(
            load_hf_model_values, layer_group_assignments, build_layer_documents),
        network=build_network_overview,
        validate_document=validate_hf_reading_document,
        validate_layer_structure=validate_hf_layer_structure,
        validate_network=validate_hf_network_overview,
        page_profile=GLM53_PAGE_PROFILE,
        export_structures=export_structures_workbook,
        export_single=export_single_structure_workbook,
        readme_single=lambda document, analysis_id: b"",
        readme_variants=readme_variants,
        identity_schema="hf_layer_structures_identity_v1",
        page_item_extra=("indexer_type",),
        identity_extra=("indexer_type",),
        manifest_group_fields=(("indexer_types", "indexer_type"),),
        manifest_fixed=(
            ("onnx", "not_implemented（GLM-5.3 适配未实现可选 ONNX；不声明支持）"),),
        effective_value_fields=(
            "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "config_head_dim_raw", "vocab_size",
            "rms_norm_eps", "hidden_act", "use_cache", "max_position_embeddings",
            "tie_word_embeddings", "q_lora_rank", "kv_lora_rank", "qk_head_dim",
            "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "attention_bias",
            "attention_dropout", "attention_scaling", "index_n_heads",
            "index_head_dim", "index_topk", "index_topk_freq",
            "index_skip_topk_offset", "rope_theta", "rope_type",
            "first_k_dense_replace", "intermediate_size", "moe_intermediate_size",
            "n_shared_experts", "n_routed_experts", "num_experts_per_tok",
            "scoring_func", "norm_topk_prob", "routed_scaling_factor", "n_group",
            "topk_group", "moe_router_dtype", "num_nextn_predict_layers",
        ),
        effective_extra_fields=("model_revision", "checkpoint_transformers_version"),
        effective_layer_fields=("layer_types", "mlp_layer_types", "indexer_types"),
        effective_parameter_records=effective_parameter_records,
        variant_effective_config=True,
        onnx_expected=None,
    )


# ---------------------------------------------------------------------------
# GLM-5.3-Flash 适配
# ---------------------------------------------------------------------------


def _glm53flash_adapter() -> _Adapter:
    from structure_reader.checks.glm53flash_hf_contracts import (
        validate_hf_layer_structure,
        validate_hf_network_overview,
        validate_hf_reading_document,
    )
    from structure_reader.extract.glm53flash_hf import build_layer_documents
    from structure_reader.extract.glm53flash_network import build_network_overview
    from structure_reader.export.glm53flash_workbook import (
        export_single_structure_workbook,
        export_structures_workbook,
    )
    from structure_reader.rules.glm53flash_hf import (
        SOURCE_FILES,
        effective_parameter_records,
        layer_group_assignments,
        load_hf_model_values,
        verify_source_fingerprints,
    )

    def readme_variants(analysis_id, shipped_structures, network, onnx_modules):
        return (
            "# GLM-5.3-Flash 分层结构浏览（整网 → 层组 → Block → 算子）\n\n"
            f"- analysis_id: `{analysis_id}`\n"
            "- 官方来源：`zai-org/GLM-5.3-Flash` revision "
            f"`{network['model'].get('model_revision')}`，model_type=`glm5_next`"
            "（text_config=`glm5_next_text`）；实现为本地官方 transformers 检出"
            "（见 `reports/source_manifest.json`）。\n"
            "- 覆盖：text_config 语言主干 token→logits（Embedding → 4 路 mHC 展开 → "
            "45 层 → hc_head 均值 → Final RMSNorm → logits slice → LM head）。"
            "三组真实层型：KDA + Dense MLP(0,1,2)、MLA/DSA + TopK MoE（3,7,…,43，11 层）、"
            "KDA + TopK MoE（其余 31 层），共 45 层；组内只保留代表层模板，"
            "循环次数来自真实层索引。\n"
            "- 入口：`viewer/structure.html`（默认只显示当前层级；整网→层组→Block→内部算子，"
            "面包屑与链接可返回上级；场景参数 seqlen/kvcachelen 在同一结构上切换）\n"
            "- 数据：`data/layer_structures.json`（每组一份参数化 Block 结构 + 场景派生）、"
            "`data/network_overview.json`（token→logits 整网路径）、"
            "`data/effective_config.json`（有效配置与逐参数来源）。\n"
            "- 表：`tables/decoder_variants.xlsx`（Prefill/Decode 两页；每页三组，"
            "seqlen/kvcachelen/B/精度/位宽/CUBE 与 VEC 吞吐为可填写示例配置，派生值由 Excel "
            "公式重算；逐算子 CUBE/VEC 分类、B/M/K/N 或 M/N、工作量与用户规则耗时 ms，"
            "层组小计与整网串行汇总；未知/非法输入显示未知，不按 0 计）。\n"
            "- 关键事实：KDA 状态固定大小（不随 kvcachelen 增长）；MLA/DSA 缓存压缩 latent "
            "[B,1,L,512]（NoPE，values 0 维）与 indexer packed [B,L,257]；"
            "k-pool=4、池 top-k + tail、K_sel=min(Lidx,2048+Lidx%4)；4 路 mHC 的 "
            "post/comb 混合不是普通残差；MoE 288 选 8 + SwiGLU clamp。\n"
            "- 构建命令：`PYTHONPATH=structure_reader/src python "
            "structure_reader/scripts/read_model_structure.py build-hf --model "
            "glm_5_3_flash_hf --all-variants --archify-layers --json`\n"
            "- 限制：只做语言主干；视觉/图像输入路径不在本票范围；MTP 层 45 不在主干；"
            "未实现可选 ONNX（显式拒绝，不静默忽略）；FP8 量化执行不分析；"
            "无真实权重、非 HF 数值等价。\n").encode("utf-8")

    return _Adapter(
        model_key="glm_5_3_flash_hf",
        sample_layer_index=0,
        supports_single=False,
        supports_onnx=False,
        source_files=SOURCE_FILES,
        verify_sources=verify_source_fingerprints,
        tool_sources=_GLM53FLASH_TOOL_SOURCES,
        onnx_tool_sources=(),
        load_values=load_hf_model_values,
        layer_documents=build_layer_documents,
        structures=_structure_builder(
            load_hf_model_values, layer_group_assignments, build_layer_documents),
        network=build_network_overview,
        validate_document=validate_hf_reading_document,
        validate_layer_structure=validate_hf_layer_structure,
        validate_network=validate_hf_network_overview,
        page_profile=GLM53FLASH_PAGE_PROFILE,
        export_structures=export_structures_workbook,
        export_single=export_single_structure_workbook,
        readme_single=lambda document, analysis_id: b"",
        readme_variants=readme_variants,
        identity_schema="hf_layer_structures_identity_v1",
        page_item_extra=("indexer_type",),
        identity_extra=("attention_type", "indexer_type"),
        manifest_group_fields=(
            ("attention_types", "attention_type"),
            ("indexer_types", "indexer_type"),
        ),
        manifest_fixed=(
            ("onnx", "not_implemented（GLM-5.3-Flash 适配未实现可选 ONNX；不声明支持）"),),
        effective_value_fields=(
                "hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "config_head_dim_raw", "vocab_size",
                "rms_norm_eps", "sub_norm_eps", "hidden_act", "use_cache",
                "max_position_embeddings", "tie_word_embeddings", "pad_token_id", "dtype",
                "q_lora_rank", "kv_lora_rank", "qk_head_dim", "qk_nope_head_dim",
                "qk_rope_head_dim", "v_head_dim", "attention_bias", "attention_dropout",
                "attention_scaling", "no_rope", "index_n_heads", "index_head_dim",
                "index_topk", "index_kpool", "index_kpool_always_select_tail",
                "linear_num_heads", "linear_head_dim", "linear_conv_kernel_dim",
                "linear_lower_bound", "hc_mult", "hc_eps", "hc_sinkhorn_iters",
                "first_k_dense_replace", "intermediate_size", "moe_intermediate_size",
                "n_shared_experts", "n_routed_experts", "num_experts_per_tok",
                "scoring_func", "norm_topk_prob", "routed_scaling_factor", "n_group",
                "topk_group", "moe_router_dtype", "topk_method", "swiglu_limit",
                "num_nextn_predict_layers", "vision_depth", "image_token_id",
        ),
        effective_extra_fields=("model_revision", "checkpoint_transformers_version"),
        effective_layer_fields=("layer_types", "mlp_layer_types", "indexer_types"),
        effective_parameter_records=effective_parameter_records,
        variant_effective_config=True,
        onnx_expected=None,
    )


def _qwen38flash_adapter() -> _Adapter:
    from structure_reader.checks.qwen38flash_hf_contracts import (
        validate_hf_layer_structure,
        validate_hf_network_overview,
        validate_hf_reading_document,
    )
    from structure_reader.extract.qwen38flash_hf import build_layer_documents
    from structure_reader.extract.qwen38flash_network import build_network_overview
    from structure_reader.export.qwen38flash_workbook import (
        export_single_structure_workbook,
        export_structures_workbook,
    )
    from structure_reader.rules.qwen38flash_hf import (
        SOURCE_FILES,
        effective_parameter_records,
        layer_group_assignments,
        load_hf_model_values,
        verify_source_fingerprints,
    )

    def readme_variants(analysis_id, shipped_structures, network, onnx_modules):
        return (
            "# Qwen3.8-Flash-Next 分层结构浏览（整网 → 层组 → Block → 算子）\n\n"
            f"- analysis_id: `{analysis_id}`\n"
            "- 官方来源：`Qwen/Qwen3.8-Flash-Next` revision "
            f"`{network['model'].get('model_revision')}`，model_type=`qwen4_exp`"
            "（text_config=`qwen4_exp_text`）；实现为本地官方 transformers 检出"
            "（见 `reports/source_manifest.json`）。\n"
            "- 覆盖：text_config 语言主干 token→logits（Embedding → PLE token id → "
            "4 行 position_ids/两套 mask → 4 路 Gated Residual 展开 → 48 层 → "
            "hyper_connection_mixer 加权 mean → logits slice → LM head）。"
            "三组真实层型：GDN + TopK MoE（35 层）、GDN + N-gram PLE + TopK MoE"
            "（layer 1）、QSA + TopK MoE（3,7,…,47，12 层），共 48 层；组内只保留"
            "代表层模板，循环次数来自真实层索引。\n"
            "- 入口：`viewer/structure.html`（默认只显示当前层级；整网→层组→Block→"
            "内部算子，面包屑与链接可返回上级；场景参数 seqlen/kvcachelen 在同一结构上切换）\n"
            "- 数据：`data/layer_structures.json`（每组一份参数化 Block 结构 + 场景派生）、"
            "`data/network_overview.json`（token→logits 整网路径）、"
            "`data/effective_config.json`（有效配置与逐参数来源）。\n"
            "- 表：`tables/decoder_variants.xlsx`（Prefill/Decode 两页；每页三组，"
            "seqlen/kvcachelen/B/精度/位宽/CUBE 与 VEC 吞吐为可填写示例配置，派生值由 Excel "
            "公式重算；逐算子 CUBE/VEC 分类、B/M/K/N 或 M/N、工作量与用户规则耗时 ms，"
            "层组小计与整网串行汇总；未知/非法输入显示未知，不按 0 计）。\n"
            "- 关键事实：GDN 状态固定（conv 40,960、recurrent 786,432 每 batch），"
            "QSA 的 K/V [B,2,L,256] + indexer key [B,L,128] 随历史增长；"
            "QSA 每 4 个可见 token 均值一块，块 top-k ≤512 + tail，K_sel 逐 query；"
            "Gated Residual 读门逐元素 [4,H]、写门每支路标量 [4]，无 mHC/Sinkhorn；"
            "网络末端没有 Final RMSNorm；MoE 512 选 10（softmax，无 correction/scale/clamp）。\n"
            "- 构建命令：`PYTHONPATH=structure_reader/src python "
            "structure_reader/scripts/read_model_structure.py build-hf --model "
            "qwen_3_8_flash_next_hf --all-variants --archify-layers --json`\n"
            "- 限制：只做语言主干；视觉/图像输入路径不在本票范围；MTP 配置声明但"
            "本地实现不执行；未实现可选 ONNX（显式拒绝，不静默忽略）；"
            "无真实权重、非 HF 数值等价。\n").encode("utf-8")

    return _Adapter(
        model_key="qwen_3_8_flash_next_hf",
        sample_layer_index=0,
        supports_single=False,
        supports_onnx=False,
        source_files=SOURCE_FILES,
        verify_sources=verify_source_fingerprints,
        tool_sources=_QWEN38FLASH_TOOL_SOURCES,
        onnx_tool_sources=(),
        load_values=load_hf_model_values,
        layer_documents=build_layer_documents,
        structures=_structure_builder(
            load_hf_model_values, layer_group_assignments, build_layer_documents),
        network=build_network_overview,
        validate_document=validate_hf_reading_document,
        validate_layer_structure=validate_hf_layer_structure,
        validate_network=validate_hf_network_overview,
        page_profile=QWEN38FLASH_PAGE_PROFILE,
        export_structures=export_structures_workbook,
        export_single=export_single_structure_workbook,
        readme_single=lambda document, analysis_id: b"",
        readme_variants=readme_variants,
        identity_schema="hf_layer_structures_identity_v1",
        page_item_extra=("ple",),
        identity_extra=("attention_type", "ple"),
        manifest_group_fields=(
            ("attention_types", "attention_type"),
            ("ple", "ple"),
        ),
        manifest_fixed=(
            ("onnx", "not_implemented（Qwen3.8-Flash-Next 适配未实现可选 ONNX；不声明支持）"),),
        effective_value_fields=(
                "hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "rotary_dim",
                "partial_rotary_factor", "rope_theta", "mrope_section",
                "mrope_interleaved", "vocab_size", "rms_norm_eps", "hidden_act",
                "use_cache", "max_position_embeddings", "tie_word_embeddings",
                "pad_token_id", "dtype", "attention_bias", "attention_dropout",
                "attention_scaling", "linear_num_key_heads", "linear_num_value_heads",
                "linear_key_head_dim", "linear_value_head_dim",
                "linear_conv_kernel_dim", "output_gate_type", "indexer_n_heads",
                "indexer_kv_heads", "indexer_head_dim", "indexer_budget",
                "indexer_compress_ratio", "indexer_block_topk", "hc_count",
                "hc_lowrank", "ple_layer_ids", "ple_embed_dim",
                "ple_conv_kernel_size", "ple_conv_dilation",
                "ple_short_conv_state_len", "ngram_size", "heads_per_ngram",
                "ngram_heads", "ngram_head_dim", "ngram_vocab_size_base",
                "ngram_vocab_divisor", "split_ngram_parts", "ngram_seed",
                "ngram_context_len", "eos_token_id", "num_experts",
                "num_experts_per_tok", "moe_intermediate_size",
                "shared_expert_intermediate_size", "norm_topk_prob",
                "mtp_num_hidden_layers", "mtp_layer_types", "vision_depth",
                "vision_hidden_size", "image_token_id", "video_token_id",
                "number_of_conv_states",
        ),
        effective_extra_fields=("model_revision", "checkpoint_transformers_version"),
        effective_layer_fields=("layer_types",),
        effective_parameter_records=effective_parameter_records,
        variant_effective_config=True,
        onnx_expected=None,
    )


def _hidream_o1_adapter() -> _Adapter:
    from structure_reader.checks.hidream_o1_hf_contracts import (
        validate_hf_layer_structure,
        validate_hf_network_overview,
        validate_hf_reading_document,
    )
    from structure_reader.extract.hidream_o1_hf import build_layer_documents
    from structure_reader.extract.hidream_o1_network import build_network_overview
    from structure_reader.extract.hf_layer_structures import build_structures_container
    from structure_reader.export.hidream_o1_workbook import (
        export_single_structure_workbook,
        export_structures_workbook,
    )
    from structure_reader.rules.hidream_o1_hf import (
        SOURCE_FILES,
        effective_parameter_records,
        layer_group_assignments,
        load_hf_model_values,
        verify_source_fingerprints,
    )

    def structures(workspace_root, model_key):
        # 两个层组是独立索引空间（decoder 0..35 / ViT 0..26），按 group 构建而不是
        # 只靠 representative_layer_index 反查。
        return build_structures_container(
            workspace_root, model_key=model_key,
            values_loader=lambda root, key: load_hf_model_values(root, key),
            groups_fn=layer_group_assignments,
            build_documents_fn=build_layer_documents,
            build_variant_fn=lambda root, group, key: build_layer_documents(
                root, group_id=group["id"], model_key=key))

    def readme_variants(analysis_id, shipped_structures, network, onnx_modules):
        return (
            "# HiDream-O1-Image 分层结构浏览（整网 → 层组 → Block → 算子）\n\n"
            f"- analysis_id: `{analysis_id}`\n"
            "- 官方来源：`HiDream-ai/HiDream-O1-Image` revision "
            f"`{network['model'].get('model_revision')}`，实现为官方 GitHub commit "
            f"`{network['model'].get('source_revision')}`（`models/qwen3_vl_transformers.py`）。\n"
            "- 范围：**一次**网络 forward —— 文本 token + 像素 patch（32×32×3）+ 时间步 "
            "→ 36 层统一 Transformer → Final RMSNorm → final_layer2 → x_pred；"
            "视觉塔 27 blocks 仅在参考图/编辑/IP 任务（K>0）执行。\n"
            "- 不分析：50/28 步采样循环、CFG 两次调用、scheduler 更新、噪声注入、"
            "像素反拼与完整出图耗时。\n"
            "- 入口：`viewer/structure.html`（默认只显示当前层级；整网→层组→Block→内部算子，"
            "面包屑与链接可返回上级；场景参数改为图像尺寸/prompt/参考图规模）。\n"
            "- 数据：`data/layer_structures.json`（两组各一份参数化 Block 结构 + 场景派生）、"
            "`data/network_overview.json`（单次 forward 主路径）、"
            "`data/effective_config.json`（有效配置与逐参数来源）。\n"
            "- 表：`tables/decoder_variants.xlsx`（**单 sheet** `Generation`；"
            "图像尺寸/prompt/参考图 patch/B/精度/位宽/CUBE 与 VEC 吞吐可填写，派生值由 Excel "
            "公式重算；逐算子 CUBE/VEC 分类与耗时、层组小计与整网串行汇总；"
            "Kref=0 时参考图/视觉塔分支整段计 0；未知/非法输入不按 0 计）。\n"
            "- 构建命令：`PYTHONPATH=structure_reader/src python "
            "structure_reader/scripts/read_model_structure.py build-hf --model "
            "hidream_o1_image_hf --all-variants --archify-layers --json`\n"
            "- 限制：只做单次 forward 的静态结构；未实现单层长页与可选 ONNX（显式拒绝，"
            "不静默忽略）；无真实权重、非数值等价；不乘采样步数。\n").encode("utf-8")

    return _Adapter(
        model_key="hidream_o1_image_hf",
        sample_layer_index=0,
        supports_single=False,
        supports_onnx=False,
        source_files=SOURCE_FILES,
        verify_sources=verify_source_fingerprints,
        tool_sources=_HIDREAM_O1_TOOL_SOURCES,
        onnx_tool_sources=(),
        load_values=load_hf_model_values,
        layer_documents=build_layer_documents,
        structures=structures,
        network=build_network_overview,
        validate_document=validate_hf_reading_document,
        validate_layer_structure=validate_hf_layer_structure,
        validate_network=validate_hf_network_overview,
        page_profile=HIDREAM_O1_PAGE_PROFILE,
        export_structures=export_structures_workbook,
        export_single=export_single_structure_workbook,
        readme_single=lambda document, analysis_id: b"",
        readme_variants=readme_variants,
        identity_schema="hf_layer_structures_identity_v1",
        identity_extra=("attention_type", "moe_type"),
        manifest_group_fields=(
            ("attention_types", "attention_type"),
            ("execution", "execution"),
        ),
        manifest_network_fields=("vision_block_count",),
        manifest_fixed=(
            ("single_forward", True),
            ("sheets", ("generation",)),
            ("onnx", "not_implemented（HiDream-O1-Image 适配未实现可选 ONNX；不声明支持）"),
        ),
        effective_value_fields=(
                "hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "intermediate_size", "hidden_act",
                "rms_norm_eps", "vocab_size", "max_position_embeddings", "rope_theta",
                "mrope_section", "mrope_interleaved", "attention_bias",
                "attention_dropout", "tie_word_embeddings", "dtype", "use_cache_config",
                "vision_depth", "vision_hidden_size", "vision_heads",
                "vision_head_dim", "vision_intermediate_size", "vision_patch_size",
                "vision_temporal_patch_size", "spatial_merge_size",
                "vision_out_hidden_size", "num_position_embeddings",
                "deepstack_visual_indexes", "vision_hidden_act", "in_channels",
                "pixel_patch_size", "pixel_in_channels", "pixel_pca_dim",
                "pixel_in_dim", "pixel_out_dim",
                "timestep_frequency_embedding_size", "timestep_token_num",
                "tms_token_id", "image_token_id", "video_token_id",
                "vision_start_token_id", "external_steps_full", "external_steps_dev",
                "external_guidance_full", "external_shift_full",
        ),
        effective_extra_fields=(
            "model_revision", "checkpoint_transformers_version", "code_repository"),
        effective_parameter_records=effective_parameter_records,
        variant_effective_config=True,
        onnx_expected=None,
    )


_ADAPTERS = {
    "deepseek_v4_flash_0731_hf": _deepseek_adapter,
    "glm_5_3_hf": _glm53_adapter,
    "glm_5_3_flash_hf": _glm53flash_adapter,
    "qwen_3_8_flash_next_hf": _qwen38flash_adapter,
    "hidream_o1_image_hf": _hidream_o1_adapter,
}

SUPPORTED_MODEL_KEYS = tuple(_ADAPTERS)


def _adapter_for(model_key: str) -> _Adapter:
    factory = _ADAPTERS.get(model_key)
    if factory is None:
        raise HfBuildError(
            f"未知模型键 {model_key!r}：HF 阅读只支持 {SUPPORTED_MODEL_KEYS}；"
            "每个模型事实由自己的适配声明，不借用其它模型的提取/规则/导出"
        )
    return factory()


# ---------------------------------------------------------------------------
# 共用步骤
# ---------------------------------------------------------------------------


def _verified_input_shas(workspace_root: Path, adapter: _Adapter) -> dict[str, str]:
    """复算已分析来源指纹并返回 SOURCE_FILES 实际 SHA256（缺失/漂移即拒绝）。"""
    adapter.verify_sources(workspace_root)
    input_shas: dict[str, str] = {}
    for key, relpath in adapter.source_files.items():
        path = workspace_root / relpath
        if not path.is_file():
            raise HfBuildError(f"HF 输入缺失: {relpath}（不下载、不伪造）")
        input_shas[key] = _sha256_file(path)
    return input_shas


def _read_source_snapshot(workspace_root: Path, adapter: _Adapter) -> dict[str, str]:
    """只读读取 SOURCE_FILES 内本地源码文本，供渲染器做源码快照。"""
    source_snapshot: dict[str, str] = {}
    for relpath in adapter.source_files.values():
        path = workspace_root / relpath
        if path.is_file() and path.suffix in (".py", ".json"):
            try:
                source_snapshot[relpath] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                pass
    return source_snapshot


def _tool_source_shas(project_root: Path, adapter: _Adapter, *,
                      onnx_modules: bool) -> dict[str, str]:
    """工具源码身份：缺失即拒绝，不填 MISSING 假身份；ONNX 工具源仅在启用时计入。"""
    relpaths = adapter.tool_sources + (
        adapter.onnx_tool_sources if onnx_modules else ())
    tool_sources: dict[str, str] = {}
    for relpath in relpaths:
        path = project_root / relpath
        if not path.is_file():
            raise HfBuildError(f"工具源缺失（不填 MISSING 假身份）: {relpath}")
        tool_sources[relpath] = _sha256_file(path)
    return tool_sources


def _validate_document(adapter: _Adapter, document: dict,
                       input_shas: dict[str, str], *, label: str | None = None) -> None:
    issues = adapter.validate_document(document, input_shas)
    if issues:
        prefix = f"{label} 检查未通过: " if label else "HF 阅读数据检查未通过: "
        raise HfBuildError(
            prefix + "; ".join(issues[:8])
            + (f"（共 {len(issues)} 条）" if len(issues) > 8 else ""))


def _validate_network_document(adapter: _Adapter, network: dict,
                               input_shas: dict[str, str]) -> None:
    issues = adapter.validate_network(network, input_shas)
    if issues:
        raise HfBuildError(
            "整网总览检查未通过: " + "; ".join(issues[:8])
            + (f"（共 {len(issues)} 条）" if len(issues) > 8 else ""))


def _load_onnx_stack():
    """导入 ONNX 工具链；依赖缺失/不可导入时具名拒绝，不降级为假图。"""
    try:
        import onnx as _onnx
        import onnxruntime as _ort
        from onnxsim import __version__ as _simver
        import ml_dtypes as _mld
        import numpy as _np
        import google.protobuf as _pb
        from structure_reader.onnx_module.build import build_all as _build_all
    except Exception as exc:  # noqa: BLE001 - 依赖缺失如实拒绝，不降级为假图
        raise HfBuildError(
            f"ONNX 依赖不可用（需项目内隔离安装 onnx/onnxsim/onnxruntime）: "
            f"{type(exc).__name__}: {exc}") from exc
    env = {
        "enabled": True, "onnx": _onnx.__version__, "onnxsim": _simver,
        "onnxruntime": _ort.__version__, "numpy": _np.__version__,
        "protobuf": _pb.__version__, "ml_dtypes": _mld.__version__,
        "opset": 18, "ir_version": 10, "dtype": "float32",
    }
    return env, _build_all


def _build_onnx_document(build_all, workspace_root: Path, project_root: Path,
                         document: dict, config: dict, *, label: str,
                         expected_modules: int | None = None) -> dict:
    """一个 reading_document 的模块级 ONNX/onnxsim 构建 + 逐模块强校验。"""
    result = build_all(workspace_root, project_root, document, config)
    for phase, modules in result["report"]["modules"].items():
        if expected_modules is not None and len(modules) != expected_modules:
            raise HfBuildError(
                f"{label}/{phase}: 模块数 {len(modules)} 与 scope {expected_modules} 不符")
        failed = [mid for mid, state in modules.items() if state.get("status") == "failed"]
        if failed:
            raise HfBuildError(f"{label}/{phase} ONNX 失败: {failed[:5]}")
        for mid, state in modules.items():
            checks = state.get("checks", {})
            if checks.get("checker") != "ok" or not str(
                    checks.get("shape_inference", "")).startswith("ok"):
                raise HfBuildError(f"{label}/{phase}/{mid}: checker/shape 未通过 {checks}")
            simplify = checks.get("simplify")
            if not isinstance(simplify, dict) or simplify.get("returned") is not True:
                raise HfBuildError(f"{label}/{phase}/{mid}: onnxsim 未返回 true {simplify!r}")
    return result


def _render_html(renderer, *args, **kwargs) -> bytes:
    """渲染入口统一把渲染错误转成具名 HfBuildError。"""
    try:
        return renderer(*args, **kwargs)
    except (DecoderPageError, StructurePageError) as exc:
        raise HfBuildError(f"HTML 渲染失败: {exc}") from exc


def _export_workbook(adapter: _Adapter, source, *, structures: bool = False,
                     network=None) -> bytes:
    """工作簿导出统一错误隔离：模型适配声明 + 唯一通用导出实现。"""
    try:
        if structures:
            return adapter.export_structures(source, network)
        return adapter.export_single(source, network)
    except ImportError as exc:
        raise HfBuildError(
            f"xlsx 导出必需依赖缺失（openpyxl）: {exc}；不降级为 CSV，不发布假完整候选"
        ) from exc
    except ValueError as exc:
        raise HfBuildError(f"xlsx 导出失败（导出器拒绝输入）: {exc}") from exc


def _export_performance_workbook(adapter: _Adapter, structures, network) -> bytes | None:
    """可选性能表导出：未声明导出器的模型返回 None，不伪造新布局。

    与 ``_export_workbook`` 同一隔离口径：缺 openpyxl 具名失败、导出器拒绝输入
    具名失败；绝不静默省略新表后仍发布“成功”候选。
    """
    if adapter.export_performance is None:
        return None
    try:
        return adapter.export_performance(structures, network)
    except ImportError as exc:
        raise HfBuildError(
            "性能表 xlsx 导出必需依赖缺失（openpyxl）: "
            f"{exc}；不降级为 CSV，也不仅遗漏 tables/decoder_performance.xlsx 后"
            "发布假完整候选"
        ) from exc
    except ValueError as exc:
        raise HfBuildError(f"性能表 xlsx 导出失败（导出器拒绝输入）: {exc}") from exc


def _source_manifest_document(*, adapter: _Adapter, source_revision: str,
                              input_shas: dict[str, str],
                              tool_sources: dict[str, str], identity: dict,
                              analysis_id: str) -> dict:
    return {
        "schema": "hf_source_manifest_v1",
        "model_key": adapter.model_key,
        "source_revision": source_revision,
        "inputs": [
            {"key": key, "path": relpath, "sha256": input_shas[key]}
            for key, relpath in adapter.source_files.items()
        ],
        "tool_sources": tool_sources,
        "identity": identity,
        "analysis_id": analysis_id,
    }


def _finalize_documents(documents: dict[str, bytes], manifest: dict) -> dict[str, bytes]:
    """统一产物装配：先给全部交付文件算 SHA256，再补 run_manifest。"""
    deliverables = {name: _sha256_bytes(data) for name, data in documents.items()}
    documents["run_manifest.json"] = _pretty_bytes({**manifest, "deliverables": deliverables})
    return documents


# ---------------------------------------------------------------------------
# 构建入口
# ---------------------------------------------------------------------------


def run_hf_build(
    workspace_root: Path,
    project_root: Path,
    layer_index: int | None = None,
    model_key: str = "deepseek_v4_flash_0731_hf",
    onnx_modules: bool = False,
    all_variants: bool = False,
    archify_layers: bool = False,
) -> tuple[HfBuildResult, dict[str, bytes]]:
    """HF 阅读产物构建：生成全部交付文件字节（不落盘，由 CLI 层 _materialize 提交）。"""
    workspace_root = Path(workspace_root)
    project_root = Path(project_root)
    adapter = _adapter_for(model_key)
    if archify_layers and not all_variants:
        raise HfBuildError(
            "分层 Archify 图只随 --all-variants 分层入口生成：单层样板没有"
            "层组/Block/子图结构，--archify-layers 不会被静默忽略；"
            "请改用 --all-variants（或在 API 中传 all_variants=True）"
        )
    if all_variants:
        return _run_variants(workspace_root, project_root, adapter,
                             onnx_modules=onnx_modules, archify_layers=archify_layers)
    if not adapter.supports_single:
        raise HfBuildError(
            f"模型 {model_key!r} 只提供分层浏览入口（--all-variants）："
            "GLM-5.3 / GLM-5.3-Flash / Qwen3.8-Flash-Next / HiDream-O1-Image 适配未实现 "
            "DeepSeek 的单层长页，不伪造第二套页面；"
            "请使用 --all-variants 构建整网→层组→Block→算子入口"
        )
    if onnx_modules and not adapter.supports_onnx:
        raise HfBuildError(
            f"模型 {model_key!r} 未实现可选 ONNX 模块：不虚构 ONNX 支持，"
            "也不静默忽略显式 --onnx 请求"
        )
    if layer_index is None:
        layer_index = adapter.sample_layer_index

    input_shas = _verified_input_shas(workspace_root, adapter)
    try:
        structure, document = adapter.layer_documents(
            workspace_root, layer_index=layer_index, model_key=model_key)
    except (ValueError,) as exc:
        raise HfBuildError(f"阅读数据构建失败: {exc}") from exc

    _validate_document(adapter, document, input_shas)
    source_snapshot = _read_source_snapshot(workspace_root, adapter)
    tool_sources = _tool_source_shas(project_root, adapter, onnx_modules=onnx_modules)

    onnx_files: dict[str, bytes] = {}
    onnx_bundle = None
    onnx_env: dict = {"enabled": False}
    if onnx_modules:
        onnx_env, build_all = _load_onnx_stack()
        config = json.loads((workspace_root / adapter.source_files["config.json"]).read_text(
            encoding="utf-8"))
        result = _build_onnx_document(
            build_all, workspace_root, project_root, document, config,
            label=f"layer_{layer_index}")
        onnx_files = result["files"]
        onnx_bundle = result["bundle"]
        onnx_env["modules"] = sum(len(modules) for modules in onnx_bundle.values())

    identity = {
        "schema": adapter.identity_schema,
        "model_key": model_key,
        "layer_index": layer_index,
        "analysis_path": "eager",
        "input_sha256": input_shas,
        "tool_sources": tool_sources,
        "onnx_modules": onnx_env,
    }
    analysis_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()

    html_bytes = _render_html(
        render_decoder_page, document, source_snapshot=source_snapshot,
        onnx_bundle=onnx_bundle, candidate_id=analysis_id if onnx_modules else None)
    xlsx_bytes = _export_workbook(adapter, structure)

    values = adapter.load_values(workspace_root, model_key=model_key)
    documents = {
        "README.md": adapter.readme_single(document, analysis_id),
        "data/reading_document.json": _pretty_bytes(document),
        "viewer/decoder.html": html_bytes,
        "tables/decoder_reading.xlsx": xlsx_bytes,
        **onnx_files,
        "reports/source_manifest.json": _pretty_bytes(_source_manifest_document(
            adapter=adapter, source_revision=values.source_revision,
            input_shas=input_shas, tool_sources=tool_sources,
            identity=identity, analysis_id=analysis_id)),
    }
    documents["data/effective_config.json"] = _pretty_bytes(
        _effective_config(adapter, values))
    documents = _finalize_documents(documents, {
        "schema": "hf_decoder_build_manifest_v1",
        "status": "success",
        "model_key": model_key,
        "analysis_id": analysis_id,
        "layer_index": layer_index,
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


def _archify_level_requests(variant_ids: list[str]):
    """DeepSeek 分层图请求（T13 行为保留）：整网 + 每组 6 层 + csa-topk raw 对照。"""
    from structure_reader.render.archify_layers import _deepseek_requests

    return _deepseek_requests(list(variant_ids))


def _archify_slug(level: str, gid: str, form: str, *, pair: bool) -> str:
    if level == "network":
        return "network"
    if pair:
        return f"{level}-{gid}-{form}"
    return f"{level}-{gid}"


def _build_archify_layers(
    workspace_root: Path,
    project_root: Path,
    structures: dict,
    network: dict,
    *,
    model_key: str = "deepseek_v4_flash_0731_hf",
    module_groups_fn=None,
) -> dict:
    """同源语义/产品事实 + 模型图声明 → 分层 Archify 图（候选根内相对路径，可离线）。

    DeepSeek 继续使用 T12 语义 bundle 的 raw/normalized Attention/KV；GLM/Qwen/
    HiDream 用同一套算子级构建器 + 各自模块声明，normalized 图附带逐条化简审计
    （本轮应用 0 条，原结构保留）。所有模型共用身份/白名单/包装页/README 代码。
    """
    from structure_reader.render.archify_layers import (
        BUILDER_VERSION,
        SCHEMA,
        generate_archify_layers,
        segment_requests,
        stable_entry,
        stable_manifest,
    )

    from structure_reader.render.diagram import module_groups as _default_module_groups

    profile = graph_profile_for(model_key)
    module_groups_fn = module_groups_fn or _default_module_groups
    is_deepseek = model_key == "deepseek_v4_flash_0731_hf"
    raw_bundles = normalized_bundles = reports = None
    if is_deepseek:
        from structure_reader.extract.deepseek_v4_semantic import (
            build_all_layer_group_bundles,
        )
        from structure_reader.semantic.rules import normalize

        raw_bundles = build_all_layer_group_bundles(workspace_root)
        normalized_bundles = {}
        reports = {}
        for variant, bundle in raw_bundles.items():
            result = normalize(
                bundle,
                facts=[tuple(fact) for fact in bundle.metadata.get("shape_facts", [])],
            )
            normalized_bundles[variant] = result.normalized
            reports[variant] = result.to_json()
    requests = segment_requests(
        model_key, structures["variants"], module_groups_fn)
    repository = (
        profile.repository.as_finalize() if profile.repository_evidence else None
    )
    generated = generate_archify_layers(
        workspace_root,
        project_root=project_root,
        network=network,
        structures=structures["variants"],
        raw_bundles=raw_bundles,
        normalized_bundles=normalized_bundles,
        normalization_reports=reports,
        requests=requests,
        repo_root=workspace_root / profile.repository.root_rel,
        repository=repository,
        model_key=model_key,
        module_groups_fn=module_groups_fn,
        allow_browser_check_unavailable=True,
        source_evidence=profile.repository_evidence,
    )
    # 交付面白名单：即使生成器/桩返回了运行字段（门禁、浏览器可用性），也不写进
    # manifest 或产品页；同一身份在 Chrome 可用/受限环境重建必须字节一致。
    manifest = stable_manifest(generated["manifest"])
    generated["files"]["data/archify/manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    entries = {
        str(entry["slug"]): stable_entry(entry) for entry in manifest["entries"]
    }
    _attach_graph_details(entries, generated)
    groups: dict[str, dict] = {}
    group_of = {str(item.get("id")): item for item in network.get("groups", [])}
    for item in structures["variants"]:
        gid = str(item["id"])
        if is_deepseek:
            groups[gid] = {
                "group": entries.get(f"group-{gid}"),
                "decoder": entries.get(f"decoder-{gid}"),
                # 化简后在前、raw 对照在后：正式阅读入口默认展示化简后子图，
                # raw 只作为对照页签，不能被“取第一个”误当成主图。
                "kv": [entries[key] for key in (f"kv-{gid}-normalized", f"kv-{gid}-raw") if key in entries],
                "attention": [
                    entries[key]
                    for key in (f"attention-{gid}-normalized", f"attention-{gid}-raw")
                    if key in entries
                ],
                "moe": entries.get(f"moe-{gid}"),
                "moe_route": entries.get(f"moe-route-{gid}"),
                "moe_expert": entries.get(f"moe-expert-{gid}"),
                "moe_shared": entries.get(f"moe-shared-{gid}"),
                "moe_combine": entries.get(f"moe-combine-{gid}"),
                "mhc": entries.get(f"mhc-{gid}"),
                "q": entries.get(f"q-{gid}"),
                "o": entries.get(f"o-{gid}"),
                "cache": [
                    entries[key] for key in (
                        f"cache-main-{gid}", f"cache-compressor-{gid}",
                        f"cache-indexer-{gid}") if key in entries
                ],
                "compressor": entries.get(f"compressor-{gid}"),
                "indexer": entries.get(f"indexer-{gid}"),
            }
            continue
        block = item["structure"]["block"]
        group = group_of.get(gid) or {
            "id": gid, "label": item.get("label"), "layer_indices": item.get("layer_indices"),
            "representative_layer_index": item.get("representative_layer_index"),
            "attention_type": item.get("attention_type"), "moe_type": item.get("moe_type"),
        }
        modules = module_groups_fn(block)
        subviews: list[dict] = []
        for spec in profile.segments(block, group, modules):
            records = []
            normalized = entries.get(
                _archify_slug(spec.level, gid, "normalized", pair=spec.pair_raw))
            if normalized:
                records.append(normalized)
            if spec.pair_raw:
                raw = entries.get(_archify_slug(spec.level, gid, "raw", pair=True))
                if raw:
                    records.append(raw)
            if records:
                subviews.append({"label": spec.page_label, "entries": records,
                                 "level": spec.level, "note": spec.note})
        groups[gid] = {
            "group": entries.get(f"group-{gid}"),
            "decoder": entries.get(f"decoder-{gid}"),
            "subviews": subviews,
            "block_note": profile.block_note,
            "group_note": profile.group_note,
        }
    panel = {
        "schema": SCHEMA,
        "network": entries.get("network"),
        "endpoints": {
            "embedding": entries.get("embedding"),
            "head": entries.get("head"),
        },
        "groups": groups,
        "tool": manifest.get("tool", {}),
        "note": (
            "同源生成：节点/边/关系由同候选的结构文档程序投影；"
            "raw 与 normalized 成对出现在 Attention 层，审计写在 candidate 卡片。"
        ),
    }
    tool = dict(manifest.get("tool") or {})
    product_sources = [
        "render/archify_layers.py",
        "render/archify_profiles.py",
        "render/archify_preview.py",
        "render/hf_structure_page.py",
        "render/hf_model_profile.py",
        "render/diagram.py",
        "render/display_names.py",
    ]
    if is_deepseek:
        product_sources.extend([
            "semantic/rules.py",
            "semantic/bundle.py",
            "extract/deepseek_v4_semantic.py",
            "render/hf_reading_page.py",
        ])
    identity = {
        "schema": "hf-archify-identity.v1",
        "model_key": model_key,
        "builder": BUILDER_VERSION,
        # 回执含时间戳：不进候选交付清单（否则同输入重建无法幂等），
        # 由证据目录在同一轮 HTML 上重跑 browser-check 绑定。
        "receipts": "evidence-dir",
        "levels": list(profile.levels),
        "repository": repository or {
            "url": profile.repository.url,
            "revision": profile.repository.revision,
            "link_mode": "local-only",
            "repository_evidence": "not-declared",
            "reason": (
                "HiDream 官方实现位于嵌套目录，不是本工作区 git top-level；"
                "不冒用根 HEAD 作为 Archify 仓库证据，文件 SHA256 见 "
                "reports/source_manifest.json 与 data/archify/sources/。"
            ),
        },
        "tool": tool,
        "graphs": {
            str(entry["slug"]): {
                "candidate_sha256": entry.get("candidate_sha256"),
                "html_sha256": entry.get("html_sha256"),
                "view_sha256": entry.get("view_sha256"),
                "relations_sha256": entry.get("relations_sha256"),
                "sources_sha256": entry.get("sources_sha256"),
                "form": entry.get("form"),
                "nodes": entry.get("nodes"),
                "connections": entry.get("connections"),
            }
            for entry in manifest["entries"]
        },
        "archify_product_sources": {
            name: _sha256_file(_PKG_SOURCE_DIR / name)
            for name in product_sources
            if (_PKG_SOURCE_DIR / name).is_file()
        },
        "tool_identity_sha256": hashlib.sha256(
            _canonical_bytes(tool)
        ).hexdigest(),
    }
    return {
        "files": generated["files"],
        "receipts": generated.get("receipts", {}),
        "runtime": generated.get("runtime", {}),
        "panel": panel,
        "identity": identity,
        "manifest": manifest,
        "entries": entries,
    }


def _attach_graph_details(entries: dict[str, dict], generated: dict) -> None:
    """把完整节点文字、全量关系与画布尺寸挂到页面清单（不写进交付 manifest）。

    交付 manifest 只保留稳定哈希；页面需要“完整文字/逐条关系/内嵌尺寸”，
    因此这里从已生成的 candidate/relations 字节解析，图本身字节不变。
    """
    details = generated.get("node_details") or {}
    files = generated.get("files") or {}
    for slug, entry in entries.items():
        entry["node_details"] = details.get(slug, [])
        candidate_rel = entry.get("candidate") or ""
        candidate_bytes = files.get(candidate_rel)
        if candidate_bytes:
            try:
                candidate = json.loads(candidate_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                candidate = None
            if isinstance(candidate, dict):
                if str(slug).startswith(("mhc-", "moe")):
                    for card in candidate.get("cards") or ():
                        if str(card.get("title") or "").startswith("参数"):
                            entry["local_parameters"] = [
                                str(item) for item in (card.get("items") or ())
                            ]
                            entry["local_parameters_title"] = (
                                "mHC 局部参数（全称/来源）" if str(slug).startswith("mhc-")
                                else "MoE 局部参数（全称/来源）")
                            entry["local_parameters_note"] = (
                                "config：hc_mult、hc_sinkhorn_iters、hc_eps；"
                                "modeling_deepseek_v4.py:926-938、976-979。"
                                if str(slug).startswith("mhc-")
                                else "仅本图实际必要参数；Hash/TopK/专家/共享专家各自裁剪。")
                            break
                components = candidate.get("components") or []
                if components:
                    component_right = max(
                        float(comp["pos"][0]) + float(comp["size"][0])
                        for comp in components
                    )
                    component_bottom = max(
                        float(comp["pos"][1]) + float(comp["size"][1])
                        for comp in components
                    )
                    # 连线标签可能被错层到组件右/下缘之外；画布必须覆盖
                    # 标签包围盒，否则嵌入 iframe 会按组件尺寸裁掉标签。
                    label_right = component_right
                    label_bottom = component_bottom
                    for conn in candidate.get("connections") or ():
                        text = str(conn.get("label") or "")
                        at = conn.get("labelAt")
                        if not text or not isinstance(at, list) or len(at) != 2:
                            continue
                        try:
                            label_right = max(
                                label_right,
                                float(at[0]) + _edge_label_box_width(text) / 2.0,
                            )
                            label_bottom = max(label_bottom, float(at[1]) + 2.0)
                        except (TypeError, ValueError):
                            continue
                    entry["canvas"] = {
                        "width": label_right + 60.0,
                        "height": label_bottom + 70.0,
                    }
        relations_rel = entry.get("relations") or ""
        relations_bytes = files.get(relations_rel)
        if relations_bytes:
            try:
                records = json.loads(relations_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                records = []
            if isinstance(records, list):
                entry["relations_all"] = [
                    {
                        "from": record.get("from"),
                        "to": record.get("to"),
                        "tensor": record.get("tensor_label"),
                        "shape": record.get("shape"),
                        "kind": record.get("kind"),
                        "label": record.get("label"),
                    }
                    for record in records
                ]


def _archify_readme_section(readme: bytes, panel: dict, manifest: dict) -> bytes:
    """候选 README 追加分层图入口（同源生成，离线可开；不影响既有段落）。"""
    lines = [
        "",
        "## 分层底层图（Archify，同源生成）",
        "",
        "本候选把整网 → 层组 → Block/Decoder → 子图（Attention/KV、Indexer、"
        "FFN/MoE、mHC/门控残差、PLE、DeepStack）的底层数据流图一并交付：图由同候选的 "
        "`data/network_overview.json`、`data/layer_structures.json` 与各模型图声明"
        "程序生成，不是手抄或独立预览集合。",
        "",
        "- 图入口：`viewer/archify/`（离线单文件 HTML，可直接打开）；"
        "图数据与完整关系：`data/archify/`。",
        f"- 图数量：{len(manifest.get('entries', []))}；"
        f"Archify CLI {((manifest.get('tool') or {}).get('version'))}，"
        f"生成器 `{(manifest.get('builder'))}`。",
        "- 每张图的关系清单（张量身份 + 全称维度 + 分支/状态）在 `data/archify/edges/`；"
        "分层页各级有进入链接与返回路径。",
        "- normalized 图附化简规则审计（逐条候选/拒绝原因）；raw 图保留源码级"
        "原始算子，二者节点/源码可联查。",
        "- 若候选含 `data/archify/sources/`（HiDream：官方实现为嵌套检出），"
        "该 JSON 给出节点→路径/行号；不冒用工作区根 HEAD 作为 Archify 仓库证据，"
        "文件哈希以 `reports/source_manifest.json` 为准。",
        "",
    ]
    network = panel.get("network") or {}
    if network:
        lines.append(f"- 整网：`{network.get('html')}`")
    endpoints = panel.get("endpoints") or {}
    for key, label in (("embedding", "Embedding 端点"),
                       ("head", "输出端算子（HyperHead → lm_head）")):
        entry = endpoints.get(key)
        if entry:
            lines.append(f"- {label}：`{entry.get('html')}`（逐算子，含就近权重/参数）")
    if endpoints.get("embedding") or endpoints.get("head"):
        lines.append(
            "- 正式阅读入口 `viewer/structure.html`：四层导航（整网 → Decoder → "
            "模块关系 → 内部算子），只提供 Prefill / Decode 两个阶段，图与参数保留 "
            "seqlen/kvcachelen 符号；页面只内嵌 Archify 原生图（`?embed=1`），"
            "不再并列旧 SVG 图。")
    for gid, group in (panel.get("groups") or {}).items():
        parts = []
        for key in ("group", "decoder", "moe", "mhc"):
            entry = group.get(key)
            if entry:
                parts.append(f"{key}=`{entry.get('html')}`")
        for entry in list(group.get("kv") or ()) + list(group.get("attention") or ()):
            parts.append(f"{entry.get('level')}-{entry.get('form')}=`{entry.get('html')}`")
        for subview in group.get("subviews") or ():
            for entry in subview.get("entries") or ():
                parts.append(
                    f"{entry.get('level')}-{entry.get('form')}=`{entry.get('html')}`")
                if entry.get("sources"):
                    parts.append(f"sources=`{entry.get('sources')}`")
        if parts:
            lines.append(f"- 层组 {gid}：" + "；".join(parts))
    lines.append("")
    return readme.decode("utf-8").rstrip("\n").encode("utf-8") + b"\n" + (
        "\n".join(lines)).encode("utf-8")


def _run_variants(workspace_root: Path, project_root: Path, adapter: _Adapter,
                  onnx_modules: bool = False, archify_layers: bool = False):
    """分层浏览入口：整网总览 + 各真实层组结构 → 分层页 + 两页 xlsx。"""
    model_key = adapter.model_key
    if archify_layers and model_key not in ARCHIFY_LAYER_MODELS:
        raise HfBuildError(
            f"模型 {model_key!r} 尚未接入分层 Archify 图（五模型声明见 "
            "render/archify_profiles.py）：不借用其它模型的分段/来源规则"
        )
    if onnx_modules and not adapter.supports_onnx:
        raise HfBuildError(
            f"模型 {model_key!r} 未实现可选 ONNX 模块：不虚构 ONNX 支持，"
            "也不静默忽略显式 --onnx 请求（DeepSeek 的 ONNX 路径未改动）"
        )
    input_shas = _verified_input_shas(workspace_root, adapter)
    try:
        structures = adapter.structures(workspace_root, model_key)
    except (ValueError,) as exc:
        raise HfBuildError(f"层组结构构建失败: {exc}") from exc
    for variant in structures["variants"]:
        _validate_document(adapter, variant["document"], input_shas, label=variant["id"])
        issues = adapter.validate_layer_structure(variant["structure"])
        if issues:
            raise HfBuildError(
                f"层组 {variant['id']} 结构检查未通过: " + "; ".join(issues[:8])
                + (f"（共 {len(issues)} 条）" if len(issues) > 8 else ""))
    try:
        network = adapter.network(workspace_root, model_key=model_key)
    except ValueError as exc:
        raise HfBuildError(f"整网总览构建失败: {exc}") from exc
    _validate_network_document(adapter, network, input_shas)
    source_snapshot = _read_source_snapshot(workspace_root, adapter)
    tool_sources = _tool_source_shas(project_root, adapter, onnx_modules=onnx_modules)

    onnx_files: dict[str, bytes] = {}
    onnx_bundles: dict[str, dict] = {}
    onnx_env: dict = {"enabled": False}
    if onnx_modules:
        onnx_env, build_all = _load_onnx_stack()
        config = json.loads((workspace_root / adapter.source_files["config.json"]).read_text(
            encoding="utf-8"))
        expected = adapter.onnx_expected or {}
        for variant in structures["variants"]:
            result = _build_onnx_document(
                build_all, workspace_root, project_root, variant["document"], config,
                label=variant["id"], expected_modules=expected[variant["id"]],
            )
            for modules in result["bundle"].values():
                for entry in modules.values():
                    base = entry.get("base", "")
                    if base.startswith("onnx/"):
                        entry["base"] = f"onnx/{variant['id']}/" + base[len("onnx/"):]
            onnx_bundles[variant["id"]] = result["bundle"]
            for name, data in result["files"].items():
                rel = name[len("onnx/"):] if name.startswith("onnx/") else name
                onnx_files[f"onnx/{variant['id']}/{rel}"] = data
        onnx_env["groups"] = {v["id"]: expected[v["id"]] for v in structures["variants"]}

    archify_files: dict[str, bytes] = {}
    archify_panel: dict | None = None
    archify_identity: dict | None = None
    archify_manifest: dict | None = None
    if archify_layers:
        try:
            archify_bundle = _build_archify_layers(
                workspace_root, project_root, structures, network,
                model_key=model_key,
                module_groups_fn=adapter.page_profile.module_groups)
        except (ValueError, OSError) as exc:
            raise HfBuildError(f"分层 Archify 图生成失败: {exc}") from exc
        archify_files = archify_bundle["files"]
        archify_panel = archify_bundle["panel"]
        archify_identity = archify_bundle["identity"]
        archify_manifest = archify_bundle["manifest"]

    identity = {
        "schema": adapter.identity_schema,
        "model_key": model_key, "all_variants": True,
        "input_sha256": input_shas, "tool_sources": tool_sources,
        "onnx_modules": onnx_env,
        "structures": _identity_structures(adapter, structures["variants"]),
    }
    if archify_identity is not None:
        identity["archify"] = archify_identity
    analysis_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    page_items = _page_items(adapter, structures["variants"])
    page_renderer = render_structure_page
    if model_key == "deepseek_v4_flash_0731_hf" and archify_panel is not None:
        # D01 正式阅读入口：只显示 Archify 图（四层导航，仅 Prefill/Decode）。
        from structure_reader.render.hf_reading_page import render_reading_page

        page_renderer = render_reading_page
    html_bytes = _render_html(
        page_renderer, network, page_items,
        source_snapshot=source_snapshot, input_shas=input_shas,
        candidate_id=analysis_id,
        onnx={"candidate_id": analysis_id, "bundles": onnx_bundles} if onnx_modules else None,
        profile=adapter.page_profile,
        archify=archify_panel)
    xlsx_bytes = _export_workbook(adapter, structures, structures=True, network=network)
    # E03：已接入新布局的模型同一份真实 structures/network 生成正式性能表；
    # 未声明的模型不产出（不借用 DeepSeek 布局，也不静默省略已声明模型的表）。
    performance_bytes = _export_performance_workbook(adapter, structures, network)
    shipped_structures = _shipped_structures(adapter, structures)
    readme_bytes = adapter.readme_variants(
        analysis_id, shipped_structures, network, onnx_modules)
    if archify_panel is not None:
        readme_bytes = _archify_readme_section(
            readme_bytes, archify_panel, archify_manifest or {})
    documents = {
        "README.md": readme_bytes,
        "data/layer_structures.json": _pretty_bytes(shipped_structures),
        "data/network_overview.json": _pretty_bytes(network),
        "viewer/structure.html": html_bytes,
        "tables/decoder_variants.xlsx": xlsx_bytes,
        **onnx_files,
        **archify_files,
        "reports/source_manifest.json": _pretty_bytes(_source_manifest_document(
            adapter=adapter, source_revision=structures["model"]["source_revision"],
            input_shas=input_shas, tool_sources=tool_sources,
            identity=identity, analysis_id=analysis_id)),
    }
    if performance_bytes is not None:
        documents["tables/decoder_performance.xlsx"] = performance_bytes
    if adapter.variant_effective_config:
        values = adapter.load_values(workspace_root, model_key=model_key)
        documents["data/effective_config.json"] = _pretty_bytes(
            _effective_config(adapter, values))
    manifest_structure = _run_manifest_structure(
        adapter, network, page_items, shipped_structures)
    if archify_manifest is not None:
        manifest_structure["archify"] = {
            "schema": archify_manifest.get("schema"),
            "builder": archify_manifest.get("builder"),
            "entry_root": "viewer/archify/",
            "data_root": "data/archify/",
            "manifest": "data/archify/manifest.json",
            "entries": len(archify_manifest.get("entries", [])),
            "levels": list(graph_profile_for(model_key).levels),
            "tool_version": (archify_manifest.get("tool") or {}).get("version"),
            "receipts": "evidence-dir（回执含时间戳/浏览器环境，不进入候选交付清单）",
            "runtime_evidence": (
                "构建时的门禁/浏览器结果只写入证据目录；"
                "HTML/manifest 只含稳定内容，跨 Chrome 可用性重建字节一致"
            ),
        }
    manifest = {
        "schema": "hf_structure_build_manifest_v1", "status": "success",
        "model_key": model_key, "analysis_id": analysis_id,
        "identity": identity,
        "structure": manifest_structure,
        "onnx": onnx_env if onnx_modules else {"enabled": False, "note": "未启用 ONNX/Netron"},
        "status_checks": {"modules_failed": 0 if onnx_modules else None,
                          "checker": "ok" if onnx_modules else None,
                          "shape_inference": "ok" if onnx_modules else None},
    }
    if performance_bytes is not None:
        manifest["tables"] = {
            "performance": {
                "path": "tables/decoder_performance.xlsx",
                "layout": "layout-v2-rework",
                "exporter": "src/structure_reader/export/deepseek_perf_layout_v2.py",
                "source": "同一份 data/layer_structures.json + data/network_overview.json",
                "scenarios": [s["id"] for s in
                              structures["variants"][0]["structure"]["scope"]["scenarios"]],
            },
            "variants_legacy": {
                "path": "tables/decoder_variants.xlsx",
                "note": "旧阅读接口，旧导出器字节保持；与性能表内容身份不同，两表并存",
            },
        }
    documents = _finalize_documents(documents, manifest)
    return (HfBuildResult(
        output_dir=project_root / "outputs" / model_key / analysis_id,
        analysis_id=analysis_id, reused=False, model_key=model_key,
        layer_index=0, all_variants=True,
        variants=tuple({"id": v["id"], "label": v["label"],
                        "layers": tuple(v["layer_indices"]),
                        "representative": v["representative_layer_index"]}
                       for v in structures["variants"])),
        documents)
