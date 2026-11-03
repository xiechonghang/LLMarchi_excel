# M2视图阶段复核：尚非最终通过

2026-10-24。f6c307d0…与41b167c5…均为不含ONNX的阶段候选，历史只读。原Harness完成四组wrapper与复用完整两表；Codex实际发现并要求修复f6c style漏`{diagram_css}`、CLI源码漏指纹/摘要仍M1、构建记录仅hint等，41b已给新实际记录与源码指纹。静态CSS与交互不能代签真实浏览器。

41b再次实际回读仍有两项必须收尾：

1. Excel展示投影将tensor.shape数组直接变成display_shape字符串，再经tensor_index重复处理；原完整导出器按数组join，导致真实I/O单元格出现`hidden_streams[[, [, ,, b, ,, a, ...]]`逐字符逗号串。参数全名单元格存在不证明真实shape正确。应保持每轴字符串数组，仅逐轴做未HTML转义的名称展开，重建index共用同一组对象、只转换一次；复核真实I/O/权重及min(...)含逗号轴不拆轴。
2. 实际554计算框类型中，HCA c_rope_gen两阶段副行均截掉最后RoPE（现`_node_sub_lines`固定[:2]）。需副行全文换行且真实高度容纳，不以title或helper正确代替可见文字。

这两处已记入后续最终M2资产收口，保当前正确数据/图/讲解与完整Excel渲染布局，不重做模型或新增布局目标。尚不交用户最终M2结论。ONNX/Netron新分支仍待实现，自动PAUSED，M3未启，不Git。
