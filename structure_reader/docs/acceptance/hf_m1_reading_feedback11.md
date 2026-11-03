# 用户阅读反馈11：结构输入至输出的数据流讲解

日期：2026-10-24。用户在当前 b81c1e8a… 的 mHC 图上反馈：结构较复杂，仅结合框图仍难形成概念。希望增加文字说明，例如“输入是什么、维度是什么，ABC 一起进入什么计算，如何逐步得到最终输出”。本轮不调整图布局或重拆算子。

在完整 Prefill/Decode 页面中，给 Decoder 总览及各现有模块配直接可见的中文讲解。先以 mHC 权重生成/流坍缩及后续残差回写贯通样板，再覆盖其余模块。说明输入来源与 shape、参数和运行时张量区别、多输入怎样共同计算、关键中间 shape、分支/汇合和输出去向；复杂模块可用少量有序步骤，不重复抄算子表。

mHC 要连接“多流→生成 pre/post/comb→pre 坍缩单流→Attention/MoE→post 展开与 comb.T 混合旧流→多流”的概念闭环。诚实区分 mHC.forward 返回的 post/comb/collapsed 与 DecoderLayer.forward 执行的残差回写。shape 与配置读同一阅读数据，源码名称可追溯；B/T 和数据相关长度不编造。Prefill/Decode 的输入、历史缓存、当前位置、闭窗及无新条目条件分别准确。

本轮仅 HTML 说明层，保原 SVG、72 ONNX、reading_document/effective_config/xlsx、原粒度、源码与图联查、本地 Netron 链接。继续原 DeepSeek Harness App / DeepSeek-V4.1-Flash 1017 High；Codex 派单验收，不直接开发。自动 PAUSED，M1 待用户整体确认，M2/M3 不启，不自动提交/推送本轮产品。不代签真实浏览器或用户阅读。

2026-10-24 工程交付：当前候选14de8a08f27a3d8f7a879b663eed7373b83a03d0bb944bb59f8cea4f1af67a51，38段说明已完成，9项受影响检查通过。原图、表格、ONNX、三数据和启动器保持；本轮静态工程通过，实际阅读待确认。详[最终复核与新版入口](hf_m1_harness_narrative11_review03.md)。本轮未提交/推送，自动PAUSED保持。
