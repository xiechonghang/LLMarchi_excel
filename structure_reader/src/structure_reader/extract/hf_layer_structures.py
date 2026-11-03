"""层组的同源结构容器（参数化 Block 结构 + prefill/decode 兼容投影）。

结构事实的唯一来源是模型适配的 ``build_layer_documents``（DeepSeek:
``extract/deepseek_v4_hf.build_layer_documents``；GLM-5.3:
``extract/glm53_hf.build_layer_documents``）：每个层组只构建一份参数化 Block 结构
（``hf_layer_structure.v1``），再投影成既有 ``hf_decoder_reading.v1`` prefill/decode
阶段。本模块负责「按真实层组归属组织各层组结构」这一**数据构造**职责，不建图、
不渲染、不做文件 I/O；页面层与导出层只消费这里的输出。

``hf_layer_structures.v1`` 容器字段：``schema`` / ``model`` / ``variants[]``，
每项含 ``id``、``label``、``attention_type``、``moe_type``、``layer_indices``、
``representative_layer_index``、``structure``（一份结构）与 ``document``
（同一结构的 prefill/decode 投影，供两表 Excel 与可选 ONNX 冻结接口使用）。
GLM-5.3 另有 ``indexer_type``（full/shared）：同一 attention_type 下由 DSA indexer
模式区分层型，不能塞进 attention_type 或 moe_type 冒充。

适配器参数：``values_loader`` / ``groups_fn`` / ``build_documents_fn`` 缺省为
DeepSeek（保持既有调用与测试行为）；构建层按模型键显式传入自己的三段函数，
仍只有这一处容器构造，不复制第二套流程。
"""
from __future__ import annotations

from typing import Callable

from pathlib import Path

__all__ = ["STRUCTURES_SCHEMA", "build_structures_container"]

STRUCTURES_SCHEMA = "hf_layer_structures.v1"


def _default_loader(workspace_root: Path, model_key: str):
    from structure_reader.rules.deepseek_v4_hf import load_hf_model_values
    return load_hf_model_values(workspace_root, model_key=model_key)


def _default_groups(values) -> list[dict]:
    from structure_reader.rules.deepseek_v4_hf import layer_group_assignments
    return layer_group_assignments(values)


def _default_build_documents(workspace_root: Path, layer_index: int, model_key: str):
    from structure_reader.extract.deepseek_v4_hf import build_layer_documents
    return build_layer_documents(workspace_root, layer_index=layer_index, model_key=model_key)


def build_structures_container(
    workspace_root: Path,
    model_key: str = "deepseek_v4_flash_0731_hf",
    *,
    values_loader: Callable | None = None,
    groups_fn: Callable | None = None,
    build_documents_fn: Callable | None = None,
    build_variant_fn: Callable | None = None,
) -> dict:
    """各真实层组各一份参数化 Block 结构（分层浏览页消费）+ 同源兼容投影。

    ``build_variant_fn(workspace_root, group, model_key)`` 用于同一模型存在**多个
    不同层组**、无法只用 ``representative_layer_index`` 区分的情形（HiDream-O1-Image
    的 36 层 decoder 与 27 个 ViT block 是两个独立索引空间）。不传时保持既有
    按代表层构建的行为。
    """
    loader = values_loader or _default_loader
    groups_of = groups_fn or _default_groups
    build_layer_documents = build_documents_fn or _default_build_documents
    values = loader(workspace_root, model_key)
    groups = groups_of(values)
    variants = []
    for group in groups:
        if build_variant_fn is not None:
            structure, document = build_variant_fn(workspace_root, group, model_key)
        else:
            structure, document = build_layer_documents(
                workspace_root, layer_index=group["representative_layer_index"],
                model_key=model_key)
        variant = {
            "id": group["id"],
            "label": group["label"],
            "attention_type": group["attention_type"],
            "moe_type": group["moe_type"],
            "layer_indices": group["layer_indices"],
            "representative_layer_index": group["representative_layer_index"],
            "structure": structure,
            "document": document,
        }
        if "indexer_type" in group:
            variant["indexer_type"] = group["indexer_type"]
        # 可选 PLE 维度：Qwen3.8-Flash-Next 的同一 attention_type 下由 PLE 区分
        # 层型；DeepSeek/GLM 没有该键时不写，保持既有 JSON 字节不变。
        if "ple" in group:
            variant["ple"] = group["ple"]
        variants.append(variant)
    return {
        "schema": STRUCTURES_SCHEMA,
        "model": variants[0]["document"]["model"],
        "variants": variants,
    }
