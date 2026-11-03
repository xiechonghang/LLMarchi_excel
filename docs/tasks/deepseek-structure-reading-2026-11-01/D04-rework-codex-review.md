# D04 技术复核通过，待用户视觉审查

2026-11-03（Asia/Shanghai）。D04/#20技术复核通过，无剩余工程阻塞；用户视觉尚未确认，不关闭Issue、不启动D05。此结论不代表DeepSeek整体/旧T14验收。

候选：`6ec5957a53a7ad82710eabf8d9bc0287b54789fe6499ef9319777102cfab5fc2`。
正式入口：`structure_reader/outputs/deepseek_v4_flash_0731_hf/6ec5957a53a7ad82710eabf8d9bc0287b54789fe6499ef9319777102cfab5fc2/viewer/structure.html`。
正式页SHA256：`09a5d7988a4aee1e956ec3872fa34c0170154009b251fa7eb6a4da057a34fc39`。

## 四项返修通过

1. 四组汇合图包含真实moe_combine/index_add，消费expert_down_out与expert_token_idx，零初始化/命中专家累加后产生routed_out，再View、再加shared。权重只在专家输出加权步使用一次；关系层单专家边shape改为num_expert_tokens×hidden_size并补dispatch→combine token_idx。
2. Hash/TopK路由节点就近显示实际score激活、选择、原始scores.gather、归一化与缩放顺序。Hash用tid2eid[input_ids]选择，TopK纠偏只选择，不参与权重；与当前源码/config对应。不以公式伪造新的底层张量。
3. MoE所有关系/路由/专家/共享/汇合图的实际candidate连线显示tensor身份与全称shape，同shape的token_idx/top_k_pos/indices/weights可区分，未靠tooltip/JSON代替。
4. 共享专家gate上界/up双侧Clamp及实际激活乘法/down_proj可见，全token独立输入；本图Lim等参数保留，无关hc_eps移除。mHC就近config/源码范围说明恢复，D01–D03职责与已接受数据流保留。

## 最终候选独立验证

- 20张MoE图实际输入、图外消费者出口、最终relations/candidate.connections与真实facts投影核对通过。汇合必需算子与专家关系边界独立检查通过，所有实际可见边带身份及shape。证据`D04-rework-codex-verify/final-moe-audit.json`。
- 双视口1440×900、1920×1080动态遍历全部71图位，804/804浏览器断言通过，0失败/运行错误，178截图。包括阶段/导航/页签、来源/参数、原生打开返回、缩放/长图滚动与实际字号/标签几何。Chrome sandbox未关闭，门槛未降。证据`D04-rework-codex-verify/page/result.json`，绑定本页SHA。
- 全部67张原生图独立重新finalize，validate/deliver/check/browser-check 67/67通过；重建HTML与交付67/67字节一致。原始回执及汇总在`D04-rework-codex-verify/native/`、`native-evidence.json`。复用脚本task字段仍写D01，实际candidate/analysis_id/page SHA/每图回执均绑定本D04候选，不冒充旧证据。
- 阅读页面59/59测试由Codex独立复跑通过。dsh全套565/565是其报告结果，未冒充Codex再次全跑。证据`D04-rework-codex-verify/tests.log`。
- 历史输出保护并集28870/28870保持，Excel与返修前f5b8候选逐字节一致，来源清单与当前源码匹配。证据`D04-rework-codex-verify/protection.json`。既有删除未恢复，未集成E01改版。
- GUI核对窗口及会话层级为正确“实施 D01 图形显示修复”，最终6ec5957a交付已完成、输入空、无停止生成按钮。

## 过程与结论边界

误派E01期间的两个文件部分修改和构建已保留，并由正确D04会话核对后继续修复；未回滚。原基线与纠正快照、部分差异、错误与纠正派单记录均在D04-execution.json及D04-codex-verify保留。最终证据绑定6ec5957a，不沿用半成品回执。

复合长阅读页通用browser-check仍仅报viewer/viewport-overflow，为此前记录的长文档工具限制；不将该命令写成通过。正式入口由专用804项真实探针验证，67张原生图严格通过原生门禁，不混淆范围。

按用户要求，人眼视觉由用户负责，Codex不逐张读取全部长截图。本聊天已提供最终入口并请求视觉确认，目前待回复。用户确认后再关闭D04、推进D05整套收口；不Git、不改自动化、不操作E01或旧T14。
