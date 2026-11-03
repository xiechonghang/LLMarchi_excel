# M2长页分层目录复核

2026-10-27；用户反馈长HTML缺少跳转目录，难以定位。原DeepSeek Harness App同会话、DeepSeek-V4.1-Flash 1017 High第44/45轮实现，Codex派单/验收。范围仅HTML导航、章节锚点及局部样式。本轮navigation_static_engineering=passed；实际浏览器阅读仍待用户确认，M2整体阅读pending、M3未启、自动PAUSED。

新入口：[带目录的四组合阅读页](../../outputs/deepseek_v4_flash_0731_hf/cfa518d528fdc7709be0643d56274f31536819de1e8e837e2e044182e6dda962/viewer/decoder_variants.html)。

## 可见变化

宽屏左侧固定目录，自身可滚动；Prefill/Decode→真实四组合→参数、总览、压缩器选择、模块图、算子流、张量索引、已有覆盖说明。分组和模块列表可展开/收起；阶段和组合有独立跳转入口。CSA内Lightning Indexer嵌套显示并定位现有锚点，HCA/SWA不造Indexer。保完整Prefill后Decode，不把目录当结构分析的新事实。

≤1100px目录放顶部并限制自身高度，右下角有持续可访问“目录”链接，页底留空间；目录为原生nav/details/summary/hash，无JS也可基本定位，支持浏览器键盘操作。正文已有图↔算子、参数/张量/源码及Netron联查保留。新目录链接沿用浅暗主题ink/accent，提供focus-visible轮廓；CSS静态声明已核，不称实际焦点/窄屏视觉通过。

## Codex复核

- 读取实际两个renderer与3项有限导航检查，复跑3项通过。产品相对9a3f仅render/m2_variants.py和decoder_page.py改动；后者只加七类章节card id，原数学和数据源未改。
- 对实际交付HTML独立解析205个目录链接，目标存在且唯一；2阶段、8组合、138模块目标按真实phase/组所属定位，4个Indexer在本组CSA目录项之下。3922个id无重复，原3862个锚点全部保留；原有href在替换新候选身份后未删除。
- 8个原组合正文文字保持，142 SVG、24完整table、626源码pre逐块相同；reading_variants.json及decoder_variants.xlsx与9a3f逐字节同，全部556个ONNX目录文件（含276原简模型）逐文件相同。本轮沿用其数值/shape结论，不重跑旧数值门禁。
- 561份交付hash、28工具源、7输入与实际内容身份匹配。两真实子进程all-variants --onnx日志argv/cwd/env/stdout/stderr/rc已核，同cfa518身份，第一次reused=false、第二次true，均rc0、562文件。
- 实际launcher逐一解析138新Netron链接，均指向当前对应sim文件，resolve-only rc0；binary SHA b7b10037e451e45f56aace3cfad3982d12589d11ffcde2dbb0d3e068efbf94e1保持。没有启动协议GUI。
- 13099旧保护加中间a200后的13661历史文件全部无漂移。旧freeze/报告保留只读，本轮需求/计划/progress授权变化另记；新身份单独冻结，不移植旧冻结身份。

证据：`.work/coordination/hf_m2_toc01/`的task/fix01、原报告、有限测试、verify_delivery.py、codex_delivery_result.json、codex_scope_builds.json、codex_netron_resolve.json、原始logs及保护表。初版a200缺少阶段/组合真跳转、Indexer入口及窄屏持续入口，有限收尾已补，不覆盖初版历史记录。

限制：browser_visual=blocked/not_verified，真实鼠标跳转/键盘焦点/窄屏滚动和Netron协议GUI未代签，未绕过file/协议安全限制；native_excel未复验，本轮Excel字节不变。无搜索/自动高亮，非本轮要求。不Git提交推送，用户删除保持，原Harness正常完成空闲，除新反馈/实际新错误不重复派单。
