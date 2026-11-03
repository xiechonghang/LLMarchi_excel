# 反馈11初版复核：保留讲解框架，补齐文字真实性

2026-10-24，候选 `094e2f391b38264b6cd6e48a6d2f7dc90ff98f2ae9666324a4edd4815cd9360e`。状态 changes_required，仅新增说明层，不是模型缺陷。Codex 已读实际 helper/页面接入/测试、mHC 样板及两次构建原始记录。

保留：两阶段模块图前的说明、CSA 下 Indexer 子块、mHC 的 pre 消费及 post/comb/collapsed 返回与后续残差联系；原图/模型/三数据保持方向正确。存在块数、关键词或链接可达不等说明准确。

具名问题：

1. `_Ctx.t` 将内部语义 id 直接作为可见名，未复用已有源码别名；三个不存在的数据名 `cache_values_post` / `cache_cumulative_length` / `cache_cumulative_length_post` 被静默印出，假对象混入正文。实际 JSON 只有 cache_cumlen，values 别名和计数后态应引用真实源码表达式诚实说明，不补假数据节点。
2. mHC 的 Nhc=4/H=4096/24、RoPE 64 写死。有限内存参数替换 Nhc=5/H=128/Rk=32 后正文仍旧值；shape 字符串未 HTML 转义。ffn_hc 输入被误写成“上一层”，实际来自本层 Attention 混合输出；“输出三组系数”与真正三个返回值混淆。学习权重及关键拆分维度也应直接读 JSON 展示。
3. MoE 专家段把 gate/up 都说成 ±Lim 截断，HF 实际 gate 仅上界、up 才双界。路由文字应明确 correction bias 只选专家，最终权重 gather 未加 bias 的 scores。专家/路由/Indexer 的关键输入、权重及中间 shape 不足，尚未实现用户所需从输入维度走到输出的可读链路。
4. Decoder 总览未新增贯通说明；需要短段接起 mHC→Attention 并列支路/缓存→残差→mHC→MoE→残差，阶段前提按参数说清。

证据 `codex_harness_narrative11_review01/findings.json`、内存参数小例 HTML；候选未被改写。`before_fix01.json` 与 2635 旧候选文件保护已保存。只回同 App1017 High 修本文新增说明/helper与有限检查，保现有正确内容；不扩图布局、模型、Excel、ONNX或旧门禁。工程、浏览器和用户阅读分开，M1 未最终确认。
