# D01 最终入口复核：退回标签布局修复

复核日期：2026-11-02；候选 `f1ea240e2c0c328b075863a963526a1e64bdb717e1584b50f3b6d36821279614`。
正式入口 SHA256：`de632e2f727e85fee5ab28671647da59715753b9018a9d7b444ec60ea9c8582d`。

## 已通过与证据边界

29/29 原生浏览器门禁已通过（D01-display-codex-verify/native/summary.json）。
最终离线入口在 1440×900、1920×1080 两种视口完成四层导航、Prefill/Decode、状态保留、参数/源码联查、33 个嵌入图位、核心长图中段及末段滚动、单开返回及键盘缩放，358/358 断言通过（D01-final-codex-verify/page-verified/result.json，102 张截图）。可见文字最小约 6.253 px；隐藏的细节标签不计入文字门禁。
首轮旧探针退出时的 ENOTEMPTY 为临时目录清理竞争，不作为产品失败；后续探针正常结束。

## 阻塞项：连线标签互相遮挡

截图目检发现并以 SVG 实际包围盒复核。加入全图库标签重叠断言后共 424 项，398 通过、26 失败、0 运行错误（D01-final-codex-verify/label-audit/result.json）。每种视口有 13 张图、66 对标签重叠；两种视口合计 26 次失败图观测。通过的交互断言不能代替视觉验收。

受影响图及每视口重叠对数：head 1；decoder-swa-hash 6；mhc-swa-hash 1；moe-swa-hash 5；decoder-csa-hash 13；mhc-csa-hash 1；moe-csa-hash 5；decoder-hca-topk 9；mhc-hca-topk 1；moe-hca-topk 5；decoder-csa-topk 13；mhc-csa-topk 1；moe-csa-topk 5。

例如 decoder-swa-hash 的 Q 与 mask 输入标签、mHC comb 与残差标签，以及底部多条输入标签重叠；head 辅助输入标签也重叠。必须修复共用布局，保留所有真实依赖、完整文字与边的关联，不能以删除边、隐藏标签、缩字或过小字号通过。

结论：AC17 显示要求未通过，D01/#17 不验收、不关票；D02–D05 不放行。新增探针不是对模型语义的完整验收，D02–D04 仍需各自实施与复核。
