# 反馈10：表格卡片边界修复复核（2026-10-21）

当前候选 `b81c1e8a7ac0dc931a9f1bed804d0b014e6ec9eb0c88c5f246b6ceec757b6a6c`。
[新版HTML](../../outputs/deepseek_v4_flash_0731_hf/b81c1e8a7ac0dc931a9f1bed804d0b014e6ec9eb0c88c5f246b6ceec757b6a6c/viewer/decoder.html)。原DeepSeek Harness App / DeepSeek-V4.1-Flash 1017 High实施，Codex仅派单/验收。

用户截图中参数表被长来源路径撑出card。本轮只有decoder_page.py增加六个表格外层table-scroll，并加局部表格样式：宽度/max-width为父内容100%、border-box、横向auto；表格auto布局、宽度受限；单元格normal换行/anywhere与来源和mono列必要断词。参数/配置选择/张量索引各阶段3表均在原card内。无overflow:hidden裁切/列隐藏/省略文本/缩字号/改值，图的滚动和画布不改。

## 实际核验

Codex `codex_harness_table10_review01/check.py` 实际rc0：HTMLParser检查全部6表祖先链，均有独立table-scroll且仍在对应card/phase内，每阶段3表；6个完整table原始markup与2877逐字节相同，所以每行/每列/完整文字、长config/HF来源路径与locator、中文说明、表达式、单元格id与href均保留。全页id原序相同且唯一、内链闭合，所有href仅36本地Netron链接candidate ID变化。直接读当前实际局部CSS声明确认容器和表格百分比宽度/横向auto/正常断行；不是只查某个CSS字符串便宣称真实浏览器矩形通过。

36SVG（含反馈09刚展开两总览）全部逐字节同2877，72原简ONNX/144目录文件、三数据JSON/config/xlsx相同，启动器源及实际二进制也由193修前freeze复算保持。2331旧候选保护0漂移。当前151交付SHA、21工具源/7输入同源；两实际CLI原始argv数组/cwd/env/stdout/stderr/rc已核，同新ID、第二次reused。

Codex只复跑5项受影响检查：单文件离线/参数联查/渲染确定性、36本地链接真实resolve与下载，均通过无skip。Harness阅读页44项原始日志rc0已读，不重复整套；不以测试数当需求进度。本轮没有模型/数学/Excel/图布局变化，不重审旧数值/图54/旧门禁，也不重新注册或触发协议。旧193freeze只有decoder_page本轮授权变化及Codex三需求/计划/progress授权文档更新，其他0漂移，旧freeze未修改。

## 验证边界

`display_static_engineering=passed`（本轮表格DOM/声明/内容/受影响集成）；`browser_css_rectangles_zoom_narrow_screen=not_verified`、`user_reading=pending`。没有安全允许的真实本地浏览器尺寸实测，本轮未尝试服务/headless/CDP/其它浏览器绕过，不把静态HTML解析或CSS声明当作浏览器CSS实测。反馈08本地协议真实点击仍blocked/user_action_required，保持未触发。M1整体阅读未确认/M2M3未启，自动PAUSED；原Harness正常完成空闲，未Git提交/恢复用户删除。

证据：本目录check.py/result.json/targeted_tests.stderr/freeze.json；codex_harness_table10_dispatch01/含用户截图、修前源码/193freeze复算和2331旧保护；Harness dpsk_harness_table10_result01.md与logs/dpsk_harness_table10_01_*原始记录。历史只读保留。
