# D03 技术复核通过，待用户视觉审查

日期：2026-11-03（Asia/Shanghai）。D03/#19工程复核通过，用户视觉尚未确认，不关闭Issue、不开始D04/D05；不代表DeepSeek整体或旧T14验收。

候选：`73034fda94f87647a270cee4e88035c8e1fee83c927dde6c3feb5f6019bc8ca7`。
正式入口：`structure_reader/outputs/deepseek_v4_flash_0731_hf/73034fda94f87647a270cee4e88035c8e1fee83c927dde6c3feb5f6019bc8ca7/viewer/structure.html`。
正式页SHA256：`83f5492ccaae8838c155113d84c65890f7ad942393d2475b83149342ccc8a362`。

## 剩余技术问题已解决

- 共享阶段生成器分别判断内部消费者与图外消费者；同一tensor的两类边同时保留。两张CSA Q图的q_a_normed出口接向Indexer，三张压缩器的cache_compressed_c_post出口保留kv_concat消费者。内部chunk/windows等不冒充图外输出。
- 独立逐阶段核对13张Q/O/compressor/Indexer图全部真实输入及图外消费出口，无遗漏；重算relations与实际交付relations完全匹配，重算candidate.connections与实际交付connections一致。证据`D03-rework03-codex-verify/stage-final-audit.json`及可复跑`audit_final.py`。
- 延续已核Attention真实block_bias输入、HCA类型/位置/None分支、SWA/CSA/HCA独立缓存、Q/O入口、缓存累计长度阅读投影与全称符号/来源。此次只有共享输出规则与对应回归两文件变化，未重做其它产品路径。

## 最终候选独立验证

- 1440×900、1920×1080实际浏览器遍历55图位，644/644检查通过，0失败/运行错误，146截图。覆盖导航/页签、两阶段、源码/参数联查、原生/封装打开返回、缩放、长图滚动、实际文字与标签几何。Chrome sandbox未关闭；证据`D03-rework03-codex-verify/page/result.json`。
- 全51张原生图重新finalize，validate/deliver/check/browser-check均51/51通过；重建HTML与交付51/51逐字节一致。证据`D03-rework03-codex-verify/native-evidence.json`及原始逐图回执。复用脚本摘要task仍写D01，但其candidate/analysis_id/page SHA/逐图来源和回执均绑定本最终D03候选，不据task字面误认旧证据。
- 阅读页面53/53测试由Codex独立复跑通过；dsh报告559/559全套，未冒充Codex再次全跑。证据`D03-rework03-codex-verify/tests.log`。
- 历史输出保护并集23225/23225匹配，Excel与8136候选逐字节一致，当前源码与来源清单匹配。证据`D03-rework03-codex-verify/protection.json`。既有删除未恢复，未改E01隔离树或Excel正式导出。
- GUI核对dsh最新rework03交付完成，当前原会话输入空、无停止生成按钮，DeepSeek-V4.1-Flash 1017 High。

## 人工与工具边界

用户已要求视觉由其人眼审查，Codex不逐张读长截图。本轮保留机器截图，不宣称全部51图已人工目检。当前聊天已提供最终入口并请求视觉确认；尚未收到确认，状态为engineering_passed_awaiting_user_visual_review。

复合长阅读页通用browser-check仍只报`viewer/viewport-overflow`，符合此前已记录的长文档工具限制；不将此命令写成通过。正式阅读页以项目专用644项真实探针验证，51张原生图严格通过原生门禁；未混淆两类范围。

用户重点看CSA的Q通路/Indexer与压缩器出口、缓存前后状态与参数，以及HCA无Indexer的区别；可使用最终正式页“内部算子”的各独立页签。视觉确认后才完成D03阶段验收并推进D04。不commit/push、不改自动化、不恢复旧T14。
