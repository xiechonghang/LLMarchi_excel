# 分层结构与性能分析：任务进度与交付

状态：T01—T11 已由 dsh 开发并通过 Codex 工程验收，11 张 GitHub Issues 均已关闭。2026-10-30 用户授权开启 [Decoder 数据流图 vNext](version-decoder-vnext-2026-10-30.md)，T12/T13 已验收关闭；T14 第二轮验收后经独立两轴复查发现 6 项问题，**验收已撤回、#15 已重开返修**，见[返修单](T14-decoder-review/round-3-findings.md)。五模型 vNext 交付待返修与再次复核；原五模型分层结构与可编辑 CUBE/VEC 粗估表见 [当前交付索引](../../../structure_reader/docs/model_delivery_index.md) 和 [T11 验收](T11-cube-vec-review/acceptance.md)。

基于已确认需求与原位重构交接单。结构阶段已验收；用户于 2026-10-29 确认 CUBE/VEC 首版性能公式，性能开发与工程验收已完成。

## T01 统一既有 HF 构建，保留单层与四变体交付

依赖：无
[完整验收任务](01.md)

## T02 DeepSeek 整网总览连接真实层组与已有 Block 详图

依赖：T01
[完整验收任务](02.md)

## T03 DeepSeek 逐级浏览同一份参数化结构

依赖：T02
[完整验收任务](03.md)

## T04 统一 Excel 导出并提供可填写的分析场景

依赖：T03
[完整验收任务](04.md)

## T05 GLM-5.3 语言主干的分层图与结构工作簿

依赖：T04
[完整验收任务](05.md)

## T06 GLM-5.3-Flash 语言主干的分层图与结构工作簿

依赖：T04
[完整验收任务](06.md)

## T07 Qwen3.8-Flash-Next 语言主干的分层图与结构工作簿

依赖：T04
[完整验收任务](07.md)

## T08 HiDream-O1-Image 单次生成网络的分层图与单页工作簿

依赖：T04
[完整验收任务](08.md)

## T09 五模型结构交付整合与替换代码收尾

依赖：T05, T06, T07, T08
[完整验收任务](09.md)

## T10 按用户公式计算 DeepSeek 算子成本与串行循环小计

依赖：T04（已完成）；CUBE/VEC 公式已确认。单算子空白模板已交付，完整 DeepSeek 粗估表已验收。
[完整验收任务](10.md)

## T11 将已确认性能公式覆盖其余模型并完成性能交付

依赖：T05, T06, T07, T08, T10
[完整验收任务](11.md)

## 执行次序与阶段交付

dsh 在同一工作区串行执行，每张任务独立交付并经 Codex 复核；依赖图表示真实前置，不要求互不依赖的新增模型人为串联。T10 的整网成本阶段在 T04 和用户公式到位后即可开始，不必等待所有模型；其空白模板阶段已先行授权。T11 已完成四模型扩展及五模型最终重建。

结构阶段完成以 T09 为界；性能阶段完成以 T11 为界。当前 CUBE/VEC 公式已确认，不再因缺公式阻塞；T10、T11 均已通过工程验收。任何未确定的运行期规模仍需标明，不伪称完整耗时。

用户已批准拆分；11 张 Issues 及原生依赖已发布并核验。T01—T09 已由 dsh 开发、Codex 复核并关闭。T10 单算子空白模板已交付，见 [模板验收](T10-template-review/acceptance.md)。T10 完整 DeepSeek 粗估已通过验收，见 [T10验收](T10-cube-vec-review/acceptance.md)；T11会话“执行 T11 四模型性能表”已完成并验收，见 [T11验收](T11-cube-vec-review/acceptance.md)。定时检查实际配置为暂停、45分钟；保持暂停，不自动恢复。检查仅在新的完成、失败或需要处理时通知。

## GitHub 任务

- [T01 统一既有 HF 构建，保留单层与四变体交付](https://github.com/xiechonghang/LLMarchi_excel/issues/2)
- [T02 DeepSeek 整网总览连接真实层组与已有 Block 详图](https://github.com/xiechonghang/LLMarchi_excel/issues/3)
- [T03 DeepSeek 逐级浏览同一份参数化结构](https://github.com/xiechonghang/LLMarchi_excel/issues/4)
- [T04 统一 Excel 导出并提供可填写的分析场景](https://github.com/xiechonghang/LLMarchi_excel/issues/5)
- [T05 GLM-5.3 语言主干的分层图与结构工作簿](https://github.com/xiechonghang/LLMarchi_excel/issues/6)
- [T06 GLM-5.3-Flash 语言主干的分层图与结构工作簿](https://github.com/xiechonghang/LLMarchi_excel/issues/7)
- [T07 Qwen3.8-Flash-Next 语言主干的分层图与结构工作簿](https://github.com/xiechonghang/LLMarchi_excel/issues/8)
- [T08 HiDream-O1-Image 单次生成网络的分层图与单页工作簿](https://github.com/xiechonghang/LLMarchi_excel/issues/9)
- [T09 五模型结构交付整合与替换代码收尾](https://github.com/xiechonghang/LLMarchi_excel/issues/10)
- [T10 按用户公式计算 DeepSeek 算子成本与串行循环小计](https://github.com/xiechonghang/LLMarchi_excel/issues/11)
- [T11 将已确认性能公式覆盖其余模型并完成性能交付](https://github.com/xiechonghang/LLMarchi_excel/issues/12)

## 可编辑单算子示例

[下载单算子性能模板](../../../outputs/01a17dc8-T10-template-20261029/matmul_performance_template.xlsx)。取 DeepSeek q_a_proj，性能 C32:C34 留空，用户自行填写；不包含整网耗时结果。

## CUBE/VEC 粗估阶段（2026-10-29 新确认）

用户已提供并确认 CUBE `B*M*K*N/4096 ms`、VEC `M*N/64 ms`，按比例不取整。T10 完整 DeepSeek 性能表已通过工程验收；T11其余四模型与五模型统一最终交付已通过验收。早期空白单算子示例继续保留。最新 execution.json/tickets.json 为实际执行状态。

## 全部任务完成（2026-10-29）

T11已通过独立工程验收并关闭 [#12](https://github.com/xiechonghang/LLMarchi_excel/issues/12)，T01—T11全部完成。相关19模块619项测试通过；23组/39个sheet场景原生重算、最终来源引用补查及五模型39产物冻结源码闭包通过。算子粗估使用用户确认基准，不代替实测。默认MoE命中数等运行期输入仍需填写，未定时不伪称完整总计。自动化配置保持PAUSED/45分钟，无活动开发任务；未提交/推送、未恢复旧会话、未归档聊天。

## Decoder 数据流图 vNext（2026-10-30，T12/T13 已验收，T14 返修中）

[Decoder 底层数据流图调整结论](../../requirements/decoder_dataflow_refinement_2026_10_30.md) 已获授权进入开发：优先读清 Decoder 内部计算；Linear 保留实例名，维度使用全称；底层采用 Archify、单向主链与紧凑旁枝；等价化简保留语义和来源。三票与 GitHub 原生依赖已发布并核验：

1. [T12 / #13](https://github.com/xiechonghang/LLMarchi_excel/issues/13)：DeepSeek 原始/化简语义图与 KV、完整 Attention 的 Archify 阶段预览。第三轮最终交付已通过验收并关闭，61项新测试、十图真实浏览器检查、61项哈希闭包及旧产物保护通过。[验收及阶段入口](T12-decoder-review/acceptance.md)；历史复核保留。
2. [T13 / #14](https://github.com/xiechonghang/LLMarchi_excel/issues/14)：DeepSeek完整Decoder分层浏览已验收关闭；原七项及FFN维度回归解决，93项独立相关测试、121项哈希核对通过，最终Excel与T11相同。[验收与新版产品入口](T13-decoder-review/acceptance.md)，历史复核保留。
3. [T14 / #15](https://github.com/xiechonghang/LLMarchi_excel/issues/15)：推广GLM-5.3、GLM-5.3-Flash、Qwen3.8-Flash-Next、HiDream-O1，完成五模型 vNext 交付与统一索引。第一轮返修已解决共享状态漏线、维度/辅助输入与源码闭合，第二轮验收曾通过；随后用户要求的独立两轴评审发现 **830 条路由箭头缺标签/形状、身份含绝对主机路径、HiDream ViT 循环数 36 vs 27、索引命令不可复现、缺图↔Excel 粒度登记、卡片仍用缩写**，验收撤回并重开 #15。返修单见[round-3-findings.md](T14-decoder-review/round-3-findings.md)，评审见[版本评审报告](../../reviews/2026-10-31-version-review.md)；待返修后再次复核。

实际会话、快照、报告与复核状态见 execution.json/tickets.json。每票完成后 dsh 停止等待 Codex 复核。Excel 公式与历史交付保留；此前暂停的自动化仍保持 PAUSED/45 分钟，不恢复旧会话、不提交推送。
