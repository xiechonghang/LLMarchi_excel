# D01 显示返修复核：已确认修复，最终入口待验收

日期：2026-11-02（Asia/Shanghai）。#17 保持 OPEN；不放行 D02–D05。

本轮复核确认整网可读性修复生效，未发现需要立即再次派发的确定产品代码缺陷。**这不是完整正式工程验收通过**：最终章节页的指定视口交互与长图滚动仍需补齐。

## 产物与比较范围

- 最终候选：`structure_reader/outputs/deepseek_v4_flash_0731_hf/f1ea240e2c0c328b075863a963526a1e64bdb717e1584b50f3b6d36821279614/`。
- 正式入口：`viewer/structure.html`，2,019,910 bytes，SHA256 `de632e2f727e85fee5ab28671647da59715753b9018a9d7b444ec60ea9c8582d`。
- 整网图：`viewer/archify/network.html`，SHA256 `e720f01aa5e7933891c27130a411fb94f79c24bb3de32666da1a293a27524c9d`。
- 固定基线：`/private/tmp/llmarchi-D01-display-baseline-r0r_jicd/manifest.json`。实际源码差异为 `archify_layers.py`、`hf_reading_page.py`、`test_deepseek_reading_page.py`；构建装配文件未变。

## 修复与自动验证

- 整网主链收回单列，位置序列/RoPE 放在辅助列，保留 19 节点与 27 条关系；只修改整网图，其余 28 图与上一候选逐字节一致。
- 原生图的整网小字号失败已消除。Codex 在正常授权的执行环境运行原生 Archify `finalize`：**29/29 的 validate、deliver、严格 check、browser-check 全部通过**；所有独立验收副本 HTML 均与最终候选字节一致。
- 逐图运行 `visual-check --require-provenance`，29/29 命令通过、来源 current，共保存 **116 张**最终候选绑定截图。保存截图本身不是人工目检通过。
- 专项测试独立重跑 **28/28 通过，无跳过**。dsh 提交的 534 项完整回归日志作为实现方证据保留，本轮没有重复全部运行。
- 123 个交付哈希匹配；22,242 项原历史保护及本轮另冻结的两个 D01 候选 248 文件均未变、无缺失；新 Excel 与上一候选逐字节一致；29 图省略号保持 0。
- 原先页签隔离、权重消费边、来源映射四项修复继续保留。

## 实际查看范围与限制

Codex 已逐张查看全部 29 图的 **1440×900、浅色、默认 READ、初始视口**截图。整网的主链回到中间，文字及默认视口比例比上一候选改善，未记录新增确定产品缺陷。

这次查看没有覆盖所有长图的下半段、所有深色/大视口截图，也没有替代最终章节入口的固定视口交互。长图截图在视口底部截到后续内容，不据此直接认定为画布裁切；必须用滚动检查区分。完整 AC18/AC19 仍未勾选。

通过 Codex 原生 `open_in_codex` 打开最终 `viewer/structure.html` 与 `#fig-network`，实际截图确认章节正文及 Archify iframe 可加载。此预览处于应用侧栏面板，没有作为 1440×900 / 1920×1080 的精确页面视口测量，不宣称四层往返、阶段切换、源码/参数联查与查看器缩放均已在真实浏览器中通过。

本轮未执行 dsh 提供的自定义 CDP 页面驱动脚本。其代码还需注意：多项交互只记录 `found/visible/headStillVisible` 等字段而未据此抛出错误，退出 0 不能独立代表全部断言通过。后续若采用该脚本，必须逐项评估实际结果及截图，不能只读完成状态。

## 证据入口

- `D01-display-codex-verify/independent-checks.json`：交付、历史保护、Excel、图库变更与测试范围。
- `D01-display-codex-verify/native/summary.json`：全图库原生门禁、来源与副本字节绑定。
- `D01-display-codex-verify/native/<slug>/visual/`：逐图视觉回执、四张 PNG、HTML 联系表。
- `D01-display-codex-verify/perceptual-review.json`：逐张已查看截图及未覆盖范围。
- `D01-display-codex-verify/page-preview/`：实际正式入口和整网 iframe 预览截图。

## 验收状态与下一步

状态为 **awaiting_final_page_review**，本轮实现方停止点已由结果报告确认；没有新的开发任务正在派发。下一步只补最终离线入口在两种指定视口下的完整显示/交互与长图滚动证据，然后决定 #17 能否关闭。不修改验收标准，不提前实现 D02–D05，不恢复自动化。
