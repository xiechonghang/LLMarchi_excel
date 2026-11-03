# D01 Codex 复核：需要返修

日期：2026-11-02（Asia/Shanghai）。Issue：#17；父规格：#16。

结论：**changes_requested，未验收，不关闭 #17，不放行 D02–D04。**
本结论依据实际源码及候选字节；未执行浏览器目检，不将静态结果称为视觉通过。

## 固定基线与产物

- 比较基线：派发前快照 `/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-D01-baseline-albcgewm/manifest.json`。HEAD 仍为 `947ae78f120279403ec0a4c539e99d3d8869e1d6`；此前未提交改动不归入 D01。
- 候选：`structure_reader/outputs/deepseek_v4_flash_0731_hf/417846d5e495d5f8c4556f148068ad66da35ecb42bf7006a61046fa95bd60698/`。
- 正式入口：`viewer/structure.html`，2,015,980 bytes，SHA256 `81da16b616ad20c21180378e3f70ca42a1723ffed940769e316a419e5c01571e`。
- 比较到四个既有源码/测试文件改变、两个新增源码/测试文件；`tickets.json` 为 Codex 协调变更。

## Standards 轴

未发现已记录编码规范的新增硬违规。以下为确定的功能错误，不是风格建议。

1. **P1：跨章节页签互相清空。** `render/hf_reading_page.py:1159` 的 `bindTabs()` 查询整页 `[data-tab]`，点击整网 `head` 后，将所有 `block-*` 与 `ops-*` 面板隐藏。`show()`（1130 行）只恢复章节，不恢复有效页签。相反方向也会隐藏整网两个端点。`reveal()`（1051 行附近）也使用全局选择器，需一起修复。按页签容器隔离状态，并保证跨层返回后有有效选中图。
2. **P1：端点权重依赖缺边。** `render/archify_layers.py:745` 先为每个 `op.weights` 建立 `aux_of`，817–819 行再据此跳过连线，将“节点已创建”误当成“消费关系已创建”。最终候选的 `n-e-embed_tokens_weight`、`n-e-hc_head_fn`、`n-e-lm_head_weight` 均有节点而无任何关联边。按 `(权重, 消费算子)` 去重，保留所有真实辅助依赖。
3. **P2：端点外部张量来源错误。** `render/archify_layers.py:699`、715 统一使用 `_sources(selected[0])`，导致输出端 `lm_head_weight`、`final_norm_weight` 等指向 HyperHead 展平/无权 RMSNorm 的 1000、993、71–72 行。应使用张量实际定义或对应消费算子的来源，并明确来源含义。

## Spec 轴

- 上述页签缺陷违反 AC05 的“不得出现空白页签”；缺边违反 AC15 的实际输入输出关系核对；错误来源违反 AC13 来源联查要求。两条复核线独立发现相同三项问题。
- **P1：公开子图仍有主图截断。** 独立扫描最终 candidate JSON：12 张 Decoder/mHC/MoE 图共 243 处 `…`。AC14 以及规格共同约束明确“完整文字表不是主图截断的替代”；#17 明确 AC14–AC22 适用于本票全部公开范围，显示验收不能推迟到 D05。后续模块语义细分仍归 D02–D04；本票公开图的文字显示修复不因此豁免。
- **验收缺口：AC18/AC19 与运行时 AC14–AC17 尚未完成。** dsh 已诚实记录 Chrome sandbox 初始化失败，无截图、无人工目检。当前 Codex 也没有可调用的浏览器/Computer Use 工具。保持未验收，不能用静态检查或旧回执代替。后续须在允许的环境补原生/嵌入、两种桌面视口、全部图及跨章节交互检查；不得关闭 sandbox 或绕过访问限制。
- 本轮未发现需要扩大到其他模型或提前实施 D02–D04 内容的理由。

## Codex 独立验证

- 新增 `tests/test_deepseek_reading_page.py`：15/15 通过，无跳过。原报告 521 项为 dsh 提交的测试证据，本轮未完整重跑。
- 123 个 manifest 交付项逐字节 SHA256 匹配；22,242 个受保护历史文件全部未变、无缺失。
- 页面 0 个 `<svg>`、33 个 iframe、阶段按钮恰为 `prefill` / `decode`。
- 最终 JSON 确认上述三个权重节点无边，并确认输出端权重来源误指。
- 以最小 DOM 桩运行实际 `bindTabs()` JavaScript，点击 `head` 后所有模块/内部通路面板均 `hidden=true`，复现页签作用域错误。此证据只验证逻辑，不是浏览器检查或视觉验收。
- 独立原始检查：`D01-codex-verify/independent-checks.json`；逻辑复现：`D01-codex-verify/tab-scope-repro.cjs`、`tab-scope-result.json`。

## 返修交付要求

修复上述三项新错误及全部公开图的文字截断，加入有意义的跨组页签/权重消费关系/来源断言；生成新候选并重新绑定证据，保留本候选及历史产物与 Excel。浏览器条件允许时补全显示检查；若仍不可用，按实保留未验收，不关闭 #17。完成后仍停待 Codex 复核。

两轴各发现三项实现问题；规格轴另有全图文字不合格和浏览器验收缺口。最严重的实现问题为跨章节空白与权重依赖缺边。
