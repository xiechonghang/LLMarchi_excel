# 09: 五模型结构交付整合与替换代码收尾

## What to build

用户通过一致入口使用五个已适配模型，找到每个模型的结构阅读产物和覆盖范围；已被替换的重复产品代码收敛，历史候选不被误删。

## Acceptance criteria

- [ ] 五个模型的身份、支持范围、静态分析边界和产物入口在交付说明中明确，不把单个样板通过写成所有模型已通过。
- [ ] 核对统一输入/图/导出路径，没有重复复制提取器、渲染器或工作簿保存流程。
- [ ] 有替代行为和调用迁移证据后清理重复实现；仍有 CLI/注册/测试调用的旧 reference/DSpark 不整包误删。
- [ ] 执行受影响的结构和入口验证，保留模型语义回归；用户阅读体验与静态工程结果分别记录。
- [ ] 提供各阶段产物目录、复用/重构/删除/新增清单和剩余限制。

## Blocked by

- https://github.com/xiechonghang/LLMarchi_excel/issues/6
- https://github.com/xiechonghang/LLMarchi_excel/issues/7
- https://github.com/xiechonghang/LLMarchi_excel/issues/8
- https://github.com/xiechonghang/LLMarchi_excel/issues/9

## 执行与阶段结果

实际产品开发由 dsh 执行，Codex 负责派单和复核。优先复用并原位重构既有工程，保护接手前未提交改动，不自动提交或推送。每票提供可打开或可验证的产物、实际测试结果、复用/重构/删除/新增清单及未验证范围；静态工程通过不代签用户阅读验收。

需求已于 2026-10-28 确认，任务拆分已获用户批准。
