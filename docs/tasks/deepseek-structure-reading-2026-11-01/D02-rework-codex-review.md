# D02 返修验收通过

日期：2026-11-03（Asia/Shanghai）。D02/#18 按本票范围验收；用户明确反馈“视觉部分没什么问题”。D03–D05 尚未交付，DeepSeek 整体与旧 T14 不因此验收。

候选：`c47633db4bec814108f664efb4a0898f862c97f49eeaa860291868fa872b2486`。
正式入口：`structure_reader/outputs/deepseek_v4_flash_0731_hf/c47633db4bec814108f664efb4a0898f862c97f49eeaa860291868fa872b2486/viewer/structure.html`。
SHA256：`5a9337a70f80e589495bdab9a9b1097ffc10839049a968db2db0f64c44d09e60`。

## 四项返修结论

1. Decoder/mHC 连线恢复“身份｜完整符号 shape”。复核四文件差异，统一 `_edge_display_text` 覆盖主链、侧链、跨秩和返回边，未以 tooltip 替代主图。
2. 四组 mHC 标签碰撞：已完成的真实浏览器探针在 1440×900、1920×1080 合计 424/424 通过，0 失败、0 运行错误；包括标签重叠及原有导航、缩放、滚动检查。未改探针阈值。
3. 全部 29 张原生图 validate/deliver/check/browser-check 通过；两张 CSA Decoder 已通过原生门禁。重建 HTML 与交付 29/29 字节一致。本次再次核对交付 HTML 哈希与这些证据一致。
4. 核对 HF modeling_deepseek_v4.py:926–984、1169–1185：mHC 返回 post/comb/collapsed；分支计算、post 展开和 comb 转置残差混合在 Decoder。图示初始列归一与 `(hc_sinkhorn_iters-1)` 次行/列归一一致；参数卡及页面提供 hc_mult/hc_sinkhorn_iters/hc_eps 及来源。当前配置 20 次，详情 19 次与当前配置相符。

## 证据和人工审查边界

复用崩溃前已完成的独立证据 `D02-rework-codex-verify/page/result.json`、`native-evidence.json`、`tests.log`，避免重复昂贵验证。阅读页面测试 41/41 通过；dsh 全套 547 项为其报告结果，未宣称本次重新运行。

用户在当前聊天确认视觉无问题，作为本轮人工阅读验收。Codex 不宣称已逐张审阅全部长图；后续由机器检查技术与几何、用户审查视觉可读性。原 AC19 中 Codex 全图逐张目检的执行方式按本次用户要求调整，截图与来源绑定继续保留。

原生证据脚本复用自 D01，其摘要 `task` 字段仍为 D01；实际 analysis_id、candidate、page SHA 与逐图回执均绑定本 D02 返修候选，已复核。不修改历史原始证据。复合长阅读页通用 browser-check 的纵向 overflow 仍保留为工具限制，不能称该命令通过；页面使用项目专用 424 项实际浏览器探针。

## 保护与来源

当前来源清单涉及的 36 项工具/图源哈希全部匹配，29 张交付图 HTML 哈希与原生证据一致。返修前候选124、D01已接受候选124、此前候选124及248文件均无变化，Excel与返修前逐字节一致。

最早保护清单22242项中22240项匹配，两个外部动态文件变化如实保留：`structure_reader/outputs/.DS_Store`（Finder元数据）与 `docs/tasks/model-structure-2026-10-28/monitor-state.json`（08:54自动监控记录）。二者不属于模型/图/Excel产品；不恢复旧值、不声称22242全不变。详细核对见 `D02-rework-codex-verify/acceptance-binding.json`。两处既有删除未恢复。

GUI 核对原 dsh 会话“实施 D01 图形显示修复”已结束D02返修、输入空、无停止生成按钮，模型 DeepSeek-V4.1-Flash 1017 High。

下一步：继续原会话派 D03/#19；E01 Excel 改版使用隔离快照，D03 不修改其源码快照/交付/状态。不 commit/push，不恢复自动化，不接旧 T14。
