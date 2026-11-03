# 反馈11最终静态复核：结构数据流讲解

2026-10-24。原 DeepSeek Harness App / DeepSeek-V4.1-Flash 1017 High 正常完成初版及同任务两次有限文字收尾。当前候选：

`14de8a08f27a3d8f7a879b663eed7373b83a03d0bb944bb59f8cea4f1af67a51`

[新版 HTML](../../outputs/deepseek_v4_flash_0731_hf/14de8a08f27a3d8f7a879b663eed7373b83a03d0bb944bb59f8cea4f1af67a51/viewer/decoder.html)。本轮仅 `display_static_engineering=passed`；用户整体阅读仍待确认。

## 可见成果与语义复核

- 两阶段各一段 Decoder 总览说明，另有 36 个模块说明（其中两处 Indexer 嵌在 CSA 内），共 38 段。模块集合与原 36 个模块锚点完全匹配。说明直接可见，不替换原图或隐藏算子流。
- mHC 连接多流输入、参数投影、pre/post/comb 三分支、pre 内部坍缩、子层处理、post 展开与 comb.T 左乘原多流的两支相加；区分本段三返回值和 DecoderLayer 后续回写。当前参数导出 4×4096=16384、24 个系数；符号 shape 与权重 shape 同见，数字由参数派生。
- 具名核对 HF :873-985 / :1161-1185、Indexer :452-456/:516-583、专家与路由 :1019-1083：mHC pre 不返回、每层两次残差、Indexer 压缩键/query/每头权重三路来源、MoE correction bias 只选专家、gate 上界与 up 双界等已正确说明。shape 的独立可见绑定重点覆盖 mHC、路由、单专家、Indexer；不将模块存在统计当作所有源码语义逐项证明。
- 文字用源码别名，保留语义 id 的原锚点联查；三个不存在的缓存 key 已删除。values 同 keys 存储与计数更新改为真实源码表达式说明。Prefill 空历史和 Decode 单 token、有效历史前提分开。

## 工程证据

Codex 读取实际新增 helper、页面接入、有限断言、最终报告、原始 argv/cwd/必要环境及 stdout/stderr/returncode；复跑本轮 9 个说明/页面/资产检查，全部 rc=0、无 skip。参数小变体、未知 key 拒绝、HTML 转义和语义断言通过；不重复旧模型/数值/Excel/全仓门禁，也未再构建旧候选。

- 36 张 SVG 与 b81c 逐字节相同，原 6 个完整 table markup 相同。
- ONNX 目录 144 文件（含 72 模型）及 reading_document/effective_config/xlsx 逐字节相同。启动器 binary SHA `859728cbcdf9102837e2e235e106f4336776e68361201fea40390fcfb6b07054` 不变。
- 151 个 manifest 交付 SHA、22 个实际工具源（含新增 helper）、7 个输入均匹配当前字节；由真实 identity 重算 analysis_id 正确。两次真实新 build rc=0、同 ID，第二次 reused=true。
- 原有和新增内链闭合，无重复 id；36 个本地 Netron 链接 resolve 到当前对应 sim，108 下载链接无断链；未实际触发协议或 GUI。
- 2787 份旧候选文件保护复算无漂移。初始193文件冻结变化仅本轮授权源/受影响测试与Codex需求、计划、进度更新；新增 helper 独立纳入实际身份。根部用户删除的两文档保持删除，未提交或恢复。
- 新正文取实际主题变量配色：浅主题正文/背景静态对比13.02、链接7.64；暗主题11.53、链接5.57。这里只核当前有限选择器的声明配对，不是浏览器真实 CSS/排版验收。

Codex 证据：`.work/coordination/hf_m1/codex_harness_narrative11_review03/{check.py,result.json,targeted_tests.stderr}`。Harness 原始记录：`logs/dpsk_harness_narrative11_fix02_*`，可见样板：`narrative11/{prefill,decode}_fix02_sample.html`。

## 范围与限制

不移植初版或旧候选的通过结论；前两次复核与原错口径证据保留。fix01 identity 的2483保护/freeze0记录不可作为该轮2635清单证据，最终已按实际2787清单复算。

`browser_visual=blocked/not_verified`、`native_excel=not_verified`、`user_reading=pending`。没有改服务/浏览器/协议/CDP绕过既有安全限制，没有代签用户对理解程度的认可。当前无额外模型/布局任务；自动 PAUSED，M1 等用户整体确认，M2/M3 未启。本轮源码与文档尚未提交/推送 GitHub。
