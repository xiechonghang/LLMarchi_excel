# 11: 将已确认性能公式覆盖其余模型并完成性能交付

## What to build

将同一公式绑定与串行汇总能力应用于其余已完成的模型结构，最终提供五模型的可编辑性能分析产物，不为每个模型再建独立性能导出器。

## Acceptance criteria

- [x] 每个模型的实际算子及数据规模映射到用户确认的公式；新算子缺公式时列明待提供，不凭名称套错公式。
- [x] 文本模型保持两页、HiDream 一页，HiDream 仅单次神经网络调用，不乘外部采样步数。
- [x] 真实循环层级、执行假设、精度与存储位宽可追溯，成本汇总没有父子重复计数。
- [x] 提供各模型工作簿、可核对计算例与限制，清楚区分结构覆盖、公式覆盖和性能估算完成状态。

## Blocked by

- https://github.com/xiechonghang/LLMarchi_excel/issues/6
- https://github.com/xiechonghang/LLMarchi_excel/issues/7
- https://github.com/xiechonghang/LLMarchi_excel/issues/8
- https://github.com/xiechonghang/LLMarchi_excel/issues/9
- https://github.com/xiechonghang/LLMarchi_excel/issues/11

## 执行与阶段结果

实际产品开发由 dsh 执行，Codex 负责派单和复核。优先复用并原位重构既有工程，保护接手前未提交改动，不自动提交或推送。每票提供可打开或可验证的产物、实际测试结果、复用/重构/删除/新增清单及未验证范围；静态工程通过不代签用户阅读验收。

## 已确认的首版性能规则（2026-10-29）

沿用用户确认 CUBE `B*M*K*N/4096 ms`、VEC `M*N/64 ms`，不向64取整。公式与公共能力已验收；dsh 已完成四模型扩展和五模型重建，Codex 已独立复核通过（619项相关回归、23组/39页原生重算场景、最终来源引用补查）。执行假设和未知输入仍明确展示，完整总计只在必要输入均有效时产生。

## T10留给本票的交付细节

- [x] 性能区横向浏览保留算子ID（启用engine时C2或等效冻结），旧无engine路径不改。
- [x] 成本列标题区分默认4096/64与表内可改吞吐，防止改值后标题歧义。

## 工程验收完成

五模型当前交付入口：`structure_reader/docs/model_delivery_index.md`。结果：`T11-cube-vec-result.md`；独立复核：`T11-cube-vec-review/acceptance.md`。保持未提交工作树、既有删除、历史候选和单算子模板；自动化 PAUSED/45分钟。
