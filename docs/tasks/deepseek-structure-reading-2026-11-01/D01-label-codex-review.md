# D01 标签返修复核及阶段验收

时间：2026-11-02T20:44:02.186694+08:00。结论：本轮返修通过，D01/#17 按本票范围验收；不代表 DeepSeek 全部完成，也不代表 T14 或 D02–D05 已验收。

正式候选：`2fa51e64bd77da2b9328cd2548322311e37ead087833750185225ba1df32c97d`；入口 `viewer/structure.html`；SHA256 `7edf05462803602bad406fa1653820f79a9690c03d8f40c6f74b3d7c84c80ea8`。

## 独立验证

- 阅读4个源文件/测试差异：统一调整连线 labelAt、画布包含标签、嵌入高度；逐图比较29张候选JSON，组件及去除标签位置后的连接内容、路由全部不变（D01-label-codex-verify/semantic-invariance.json）。本轮没有删边、改形状、隐藏标签或缩字。
- 原只读最终入口探针未经修改，对1440×900、1920×1080各完成212项，合计424/424通过，0错误；33嵌入图位覆盖，13张先前失败图的标签重叠清零，可见最小投影字号约6.253px。阶段切换、跨层路由、参数/源码联查、查看器缩放、单开/返回、核心长图中末段滚动均通过。证据 D01-label-codex-verify/page/result.json（绑定页面与探针SHA），102张实际浏览器截图。
- 全部29张图重新 finalize：validate/deliver/check/browser-check 29/29通过；重新生成HTML与交付全部逐字节一致，来源sidecar与候选匹配（D01-label-codex-verify/native-evidence.json）。Chrome sandbox保持开启，没有关闭origin保护。
- 实际逐张查看29张唯一图1440视口完整嵌入截图（D01-label-codex-verify/gallery.json）；额外查看1440 decoder-bottom、mhc-middle、attention-middle及1920 network-bottom、moe-bottom、cache-bottom。原先平行/汇入/底部标签已错层，未见新遮挡、边界裁切或远离关联线。102张截图没有全部逐一目检，不宣称所有交互状态均被人工查看。
- Codex重跑阅读页面测试34/34通过（tests.log）；dsh报告17文件540项通过，其全套结果见D01-label-verify/tests.log，未冒充Codex再次全跑。
- 独立哈希核对22242历史产物、248前期候选文件、124个派发前当前候选文件，全部不变；新候选Excel与上一候选逐字节相同；两处既有删除保持（D01-label-codex-verify/protection.json）。
- 可见GUI核对“实施 D01 图形显示修复”最终回复停待复核，输入框空、无运行停止按钮（/private/tmp/D01-label-review-dsh-current.png）。

## 工具结果的限制

dsh自身环境仍不能启动浏览器，本轮由Codex授权环境补跑；没有沿用旧HTML浏览器通过记录。
Archify browser-check对复合structure.html返回1，唯一类错误为纵向文档滚动超出单屏；它未识别该自定义阅读页的长文档布局。保留原回执，不改工具、不隐藏滚动。正式规格明确要求页面长图滚动，因此最终入口以项目专用真实浏览器探针（424项）及实际滚动截图验收；29张原生Archify图仍严格使用原生门禁并全部通过。不得把复合页原生命令写成通过。

## 后续边界

D01交付正式入口、符号两阶段与整网端点，以及共享显示修复；可复用内部子图尚需D02的Decoder/mHC、D03的Attention/缓存、D04的MoE语义与分层改进；D05做最终整套交付及用户review。已看见的内部旧抽象图不能因此被称为最终需求已完成。后续仅派发D02，完成停待复核；共享布局所有显示门禁必须保留。
