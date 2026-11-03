# D01 返修复核：四项修复确认，显示验收仍未通过

日期：2026-11-02（Asia/Shanghai）。Issue #17 保持未验收，D02–D05 未放行。

## 固定比较范围

首轮候选 `417846d5e495d5f8c4556f148068ad66da35ecb42bf7006a61046fa95bd60698` → 返修候选 `1bc0ea5cfffadc870c53296d17a73e17a35da37f44a68dfaa2889b4ce5340e80`。
源码比较从派发前快照及已保存的首轮精确 diff 重建首轮版本，再与本轮三个实际修改文件比较，避免混入原有 dirty tree。

本轮正式入口 SHA256 `fbc6306b33b68edb49cff97ffe47932d8c25cc928731a7b82a585bd861d274ef`；整网原生图 SHA256 `705df05864f25a5327bb03be57b8b95de487ded4bea0003f306d849b04ebe9e4`。

## 已确认修复

- 页签具有独立 `data-tab-group`，切换与 reveal 按组操作，进入章节恢复有效选中状态。新增测试实际执行相关 JS 逻辑；浏览器交互验收仍另列。
- `embed_tokens_weight`、`hc_head_fn`、`lm_head_weight` 已分别连到正确消费算子；共六个端点权重的依赖关系逐项核对。
- 外部张量按生产/消费身份提供来源；输出端权重不再统一指向 HyperHead 第一操作。
- 29 张交付图的节点及连线 JSON 文字中 `…` 共 0 处，上一轮 243 处已消除。

## Standards 轴

上述页签和权重依赖修复通过源码、交付关系及针对性测试复核。未以风格偏好新增返修项。规范轴复核者没有形成额外独立检查证据，未发现新的确定问题；不能据此额外宣称全面代码复核通过。规格轴复核引用本报告的 root 实测，结论仍为未验收。

## Spec 轴：仍需完成的事项

**P1：整网图在默认小桌面视口下连线文字不满足可读性要求。**

Codex 对全部 29 张最终原生 HTML 执行了 Archify `browser-check`：**28 通过，1 失败（network）**。1440×900 明暗主题中，`batch_size,seqlen,hidden_size` 的实际投影字号为 **5.602497px**，低于 6px 下限，涉及边 `f-0-n-net-embed-n-net-mask-build`。问题来自整网图的布局/画布与关系标签共同作用，入口为 `render/archify_layers.py` 的整网图构建、`_complete_sizing` 与 `build_level_candidate`。不能只依据节点主字号估算 10px 以上来宣称全图文字可读。

同一视口下页面高 1402px，浏览器报告 `unexpected-overflow`；1920×1080 等较大视口报告 `readable-vertical-scroll`。**纵向滚动本身不是缺陷**，规格允许可读的长图滚动；先修复可读性和布局，再检查其是否符合 Reader 的可读滚动契约，不要求机械压入 900px，也不得裁剪、隐藏溢出或缩小字体来凑通过。

Codex 从同一候选 JSON 生成独立验收副本，其 HTML 与交付整网图逐字节一致，严格 artifact check 通过、浏览器检查失败。随后原生 `visual-check` 保存了四张截图。已实际查看 1440×900 与 2048×1320 的浅色截图：下半部输出主链偏在右侧，左下有大块空白；连线文字明显细小。需在保留真实节点、连接、来源与全称符号的前提下调整生成器布局。

**验收缺口：最终离线章节页及全部图的逐张目检尚未完成。**

本轮已完成全部原生图的自动浏览器测量，仅对整网保存四张截图并查看其中两张。不能据此称 AC18/AC19 或所有原生/嵌入交互验收通过；后续需补两种规格视口下的章节、页签、跨层返回、长图滚动、查看器缩放和原生单开，并对最终全部图保留截图、逐张查看。其他 28 图的自动通过也不是人工视觉通过。

## 环境与证据

默认 Codex 执行环境也遇到 Chrome DevTools 初始化失败。经正常 `require_escalated` 授权，在 Codex 文件系统沙箱外运行原生 Archify 命令后，Chrome 自身 sandbox 保持启用，浏览器测量和截图均可执行。未使用 `--no-sandbox`，未改浏览器安全设置、Archify 工具或候选 HTML。dsh 原环境失败记录继续保留。

- 独立字节与依赖核对：`D01-rework-codex-verify/independent-checks.json`。
- 全图库浏览器结果：`D01-rework-codex-verify/browser-gallery/summary.json`，逐图原始回执在同目录各 slug 子目录。
- 同字节整网验收副本及来源回执：`D01-rework-codex-verify/network-capture/`。
- 整网截图/视觉回执：`D01-rework-codex-verify/network-capture/visual/`。
- 默认受限环境的失败证据：`D01-rework-codex-verify/network-browser-sandbox-unavailable.json`。

## 独立验证及停止点

- 新增/扩展阅读页测试 **21/21 通过，无跳过**；dsh 的 527 项完整回归日志已读取，本轮未重复全跑。
- 本候选 123 个交付哈希匹配；首轮候选 123 个交付哈希也匹配。
- 22,242 个历史保护项未变、无缺失；本候选 Excel 与首轮逐字节一致。
- 阶段按钮恰为 Prefill/Decode；正式入口 0 个旧 SVG、33 个 iframe。
- dsh 可见会话“实施 D01 图形显示修复”已停止并显示返修报告，与用户反馈一致。

保留本轮四项修复，继续针对整网可读性/布局及最终显示证据推进。#17 不关闭，后续模型及 D02–D05 不提前实施。

## 后续派发

已通过安装版 dsh 可见界面，在原会话“实施 D01 图形显示修复”发送 `D01-display-handoff.md`。已看到消息气泡、输入框清空、思考状态和停止按钮；本轮任务为整网可读性修复与最终显示证据补全，结果报告约定 `D01-display-result.md`。此派发不表示 #17 已验收。
