# Archify：可复用的图形交付工具

2026-10-29：用户要求评估把可视化任务交给新安装的 Archify，并允许 dsh 使用。
已读取安装版 skill 3.2；本地 doctor 通过。用户随后确认：dsh 对话中可以显式加入 `/archify` 提示其调用。此为用户确认的调用入口；本工程模型数据的 Archify 适配和端到端图形交付尚未完成，不要将可调用写成已接入产品。

## dsh 调用方式

后续适合的图形子任务，在 dsh 派单消息中显式加入 `/archify`，并给出输入数据、范围及验收要求；这不代替开发任务本身的验收。

当前包位于 Codex 的个人技能目录。必要时 dsh 可显式读取并遵守：
`/Users/xiechonghang/.codex/skills/archify/SKILL.md`。

其 CLI 是本地 Node 程序，不依赖 Codex 专有 MCP：

```sh
/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  /Users/xiechonghang/.codex/skills/archify/bin/archify.mjs doctor
```

具体出图按 SKILL.md 的类型路由、schema、finalize 和交付约定执行。不自行安装依赖、更新技能或更改全局代理配置。用户已确认的显式提示写法为 `/archify`；不要混写为未经确认的 `$archify` 自动触发。若某会话未加载技能，可提供上述绝对路径作为后备。

## 适合分担的任务

- T09：工程的“来源/配置→同源结构→图与Excel”架构/数据流图，五模型覆盖说明及交付讲解图；只记录已实现行为，规划单独标示。
- T07/T08 及其他模型：L0模块总览、L1层组、局部Block机制讲解的展示候选；采用 architecture/tree/dataflow 等合适类型。
- T10/T11 之后：从已计算并确认的串行理论耗时生成 waterfall。数据必须来自现有公式结果，明确“理论估算”；未提供公式前不造耗时或画成实测。
- 可复用其主题、搜索/聚焦、路径探索、SVG/图片导出及图形/浏览器检查，降低通用展示开发量。

## 与现有工程的接缝

从已有 `network_overview.json`、`layer_structures.json` 及模型 profile 生成 Archify candidate；保留稳定算子/张量/层组 ID 与证据。只增加展示数据转换，不再次从源码推导一份模型结构。首先做小范围可替换展示候选，再根据实际验证决定是否替换对应旧展示代码；未完成调用迁移不能删除旧入口。

Archify 的 Reading Depth/Focus 不等于本工程的 L0–L3 真实层级路由。逐级进入/返回、符号shape、场景/分支、源码/参数/张量联查仍需适配和验收，不能仅凭泛化 Viewer 功能认定满足需求。

保留模型专用源码分析/语义契约、场景求值、唯一 Excel 导出/公式重算和成本汇总。Archify finalize 的布局/来源/浏览器检查不能证明模型数学、shape或性能公式正确，不替代独立验收。

仓库图证据按技能要求绑定已提交版本。当前大量未提交工作树内容不能冒充 HEAD 对应源码；不为出图自动提交。区分模型上游固定来源、当前候选产物快照和本仓库已提交实现，无法满足来源门禁时如实记录，不伪造通过。

当前T06先完成已派发的四项P2修复；本说明不授权在返修中替换渲染器或增加新产品范围。后续派单可引用本文件，把适合的展示子任务交给 Archify。
