"""模块级真实 ONNX 图（反馈06；只依赖 onnx，不在 import 时加载工具链其它包）。

- 只做"源码事实 → 标准 ONNX 节点"的显式直译，不建通用 AST/SSA 编译器；
- 权重一律 graph input（不写实值），cache 拆 pre/post tensor；
- 由 build_hf_decoder 在显式 ONNX 选项下调用，未启用时本包不被导入。
"""
from __future__ import annotations

__all__ = ["build_all", "proto_display_graph"]


def build_all(*args, **kwargs):
    from structure_reader.onnx_module.build import build_all as _build_all
    return _build_all(*args, **kwargs)


def proto_display_graph(*args, **kwargs):
    from structure_reader.onnx_module.view import proto_display_graph as _v
    return _v(*args, **kwargs)
