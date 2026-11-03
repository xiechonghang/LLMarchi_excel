# Decoder 数据流图 vNext（2026-10-30）

用户已授权按 [调整结论](../../requirements/decoder_dataflow_refinement_2026_10_30.md) 开启开发，仍由 dsh 实施、Codex 派单和验收。先 DeepSeek，再推广其余四模型；T01—T11 的既有交付保持。

1. [T12：语义图与可追溯等价化简](12.md)，[GitHub #13](https://github.com/xiechonghang/LLMarchi_excel/issues/13)，无前置。交付 DeepSeek raw/normalized 数据和 KV/Attention Archify 阶段预览。
2. [T13：DeepSeek 产品分层浏览接入](13.md)，[GitHub #14](https://github.com/xiechonghang/LLMarchi_excel/issues/14)，依赖 T12 验收。交付完整 Decoder 的可阅读新入口。
3. [T14：四模型推广与五模型交付](14.md)，[GitHub #15](https://github.com/xiechonghang/LLMarchi_excel/issues/15)，依赖 T13 验收。交付统一五模型新图与索引。

一张票完成后停止，Codex 复核后再派下一票；不能仅凭报告存在认定验收通过。各阶段均给出可查看产物、测试证据、未知/限制和保留清单。实际派发状态见 `execution.json`，GitHub 对应票号与依赖见 `tickets.json`。

本版本落实：具名 Linear、全称维度、Archify 底层图、单向主链、紧凑旁枝及有条件的等价化简。用户本次开发授权覆盖任务单列明的保守白名单；无法证明就保留，不要求任意符号求解器。明确禁止任意计算融合、精度变化、权重数值分析或缓存语义省略。

复用现有事实与适配/构建/详情/导航/导出；可增加语义投影以保持旧事实与 Excel 不变。Archify 接入是渲染适配，不是第二套模型事实。不要改安装技能包、硬编码个人路径或为每模型手抄图；本机技能入口可配置/发现，工具和语义规则版本参与产物身份。

Excel 性能内部计算仍留待用户后续讨论；本版本不更改既有 CUBE/VEC 公式、收费或汇总，不把结构简化宣称为设备 kernel 融合。保留所有旧候选、脏改动、既有删除、已认可讨论稿和单算子模板；不 commit/push、不恢复旧会话、不恢复此前暂停的自动化、不归档聊天。
