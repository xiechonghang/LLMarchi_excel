# D01 结果报告：DeepSeek 正式阅读入口与整网端点

日期：2026-11-02（Asia/Shanghai）。范围：只做 D01 / GitHub #17；未继续旧 T14、
未执行 D02–D05、未关票、未提交/推送、未恢复自动化。状态：**实施完成，停待 Codex 复核**。

## 1. 正式新候选与入口

| 项 | 值 |
| --- | --- |
| 候选根 | `/Users/xiechonghang/Workspace/LLMarchi_excel/structure_reader/outputs/deepseek_v4_flash_0731_hf/417846d5e495d5f8c4556f148068ad66da35ecb42bf7006a61046fa95bd60698` |
| analysis_id | `417846d5e495d5f8c4556f148068ad66da35ecb42bf7006a61046fa95bd60698` |
| **正式入口** | `viewer/structure.html`（2,015,980 bytes，sha256 `81da16b616ad20c21180378e3f70ca42a1723ffed940769e316a419e5c01571e`） |
| 图入口 | `viewer/archify/`（29 张原生 Archify HTML） |
| 图数据 / 全量关系 | `data/archify/`、`data/archify/edges/` |
| 整网 / 端点 | `viewer/archify/network.html`、`embedding.html`、`head.html` |
| 交付清单 | `run_manifest.json`（123 个交付文件）、`reports/source_manifest.json` |
| 候选 README | 已追加“正式阅读入口”与两个端点图入口 |

构建命令（文档化、可复现；不实例化模型、不读取真实权重）：

```bash
PYTHONPATH=structure_reader/src \
  /Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  structure_reader/scripts/read_model_structure.py \
  build-hf --model deepseek_v4_flash_0731_hf --all-variants --archify-layers --json
```

工具身份随候选记录：Archify 3.2.0、Node v26.10.0、生成器 `t14-archify-layers/5`；
`render/hf_reading_page.py` 已纳入 `archify_product_sources`，页面字节进入候选身份。
正式阅读入口只在 `--archify-layers` 构建中启用（无图清单时仍走旧分层页，不伪造空图）。

### 证据索引

| 证据 | 位置 |
| --- | --- |
| 接手核对 / 基线差异 | `D01-verify/handoff-check.md` |
| 受保护产物、源码差异、图库文字/几何、页面契约 | `D01-verify/display-audit.json`、`display-audit.md` |
| 受影响图清单与显示口径 | `D01-verify/affected-graphs.md` |
| 测试输出（17 文件 / 521 项） | `D01-verify/tests.log` |
| 原生 finalize / check / 浏览器原始错误 | `D01-verify/native-summary.md`、`native-evidence.json`、`native/<slug>/` |
| 原生/嵌入目检记录 | `D01-verify/visual-review.md` |
| 复现脚本 | `D01-verify/collect_evidence.py`、`native_evidence.py`、`README.md` |

## 2. #17 逐项完成情况

| #17 验收项 | 结果 | 证据 |
| --- | --- | --- |
| 正式构建生成新候选及来源记录，不实例化模型或读权重 | **通过** | 构建 exit 0；`run_manifest.json` 身份含 29 图哈希、工具/源码 SHA；构建只用标准库静态读取 |
| 可见阶段恰为 Prefill/Decode，图与参数保留 seqlen/kvcachelen 符号 | **通过** | 页面 `data-stage-btn` 恰 2 个（prefill/decode）；旧 5 类数值场景标记 0 处；`seqlen` 2036 次、`kvcachelen` 209 次；不强制 Decode 长度 |
| 正式入口不重复呈现旧 SVG 与 Archify；旧文件/候选保留 | **通过** | 页面内 `<svg>` 0 个、旧图容器 0 个；33 个内嵌图位（29 张唯一图，部分子图在两层复用）全部是 Archify `?embed=1`；22242 个历史产物逐字节未变 |
| 整网对应 ForCausalLM；循环次数/层序来自 config；模板不表示共享权重 | **通过** | 入口类 `DeepseekV4ForCausalLM`；config 43 层 = `layer_order` 43；页面与图内均写明“模板复用不代表共享权重” |
| Embedding 与输出端有具名算子、形状及就近参数；HyperHead 不冒充单算子；输出位置由 logits_to_keep 约束 | **通过** | 新增 `embedding`（5 节点/3 关系）与 `head`（14/12）；HyperHead 拆为展平+归一 / 门控投影 / sigmoid 门控 / 加权求和 4 个具名算子；`Lk`、`Ssel` 就近参数与 `logits_to_keep` 说明在页内 |
| 四层导航接入真实内容；返回、单开、源码与参数联查离线可用；阶段性范围如实声明 | **通过（静态）** | 四层 section 1/4/4/4；路由 595、锚点 1901、相对文件 168，失效/缺失 0；外链 0；每图有“子图页（带返回）/原生图新标签”；算子卡 277 个、源码面板 626 个；页首声明 D02–D04 未交付 |
| 对本票新增及共享页面改动影响的全部图做完整文字、来源/形状、路径边界、标签关联及原生/嵌入检查 | **部分完成** | D01 图（整网/端点/四个层组上下文）主图截断 0；页面完整文字 696 行、`…` 0；29 图 Archify validate/deliver/check 全过、重建 HTML 29/29 与交付字节一致；**浏览器/目检不可用**（见 §5），复用子图 243 处既有主图截断如实登记 |
| Excel 与历史产物逐字节保护；证据绑定最终候选；失败/跳过/未目检分别列出 | **通过** | 22242/22242 未变、0 缺失/0 变更；既有两处删除保持；新候选 Excel 与上一候选逐字节一致；证据目录按状态分类 |

父规格 AC01–AC07、AC13 在本票公开范围落实；AC14–AC17 的静态部分已核对，
AC18/AC19 因浏览器环境不可用而未执行；AC20 对原生图完成字节绑定，浏览器部分未完成；
AC21/AC22 已按实记录。AC08–AC12 属于 D02–D04，本票未提前实现。

## 3. 复用 / 扩展 / 新增

**直接复用（未改生成逻辑）**

- `build_hf_decoder.run_hf_build` 与 `--all-variants --archify-layers` 正式构建链路；
- 整网/层组/Decoder/KV/Attention/MoE/mHC 的 Archify 图构建器与语义 bundle；
- 层组结构、网络文档、语义/形状来源、候选身份与原子交付、Excel 导出；
- 既有算子卡、参数行、张量行、源码快照与面包屑/路由机制。

**扩展**

- `render/archify_layers.py`：新增端点图构建器（只从整网 path 取算子/张量/权重，
  不重写一份结构）；新增“完整文字”节点几何（按文字预算加宽，仅 D01 的
  整网/端点/层组图启用）；层组图不再用省略号压缩 20 个层索引。
- `render/detail_render.py`：算子卡“回图”链接可替换（阅读页没有旧 SVG）。
- `build_hf_decoder.py`：面板增加端点图；为页面附带完整节点文字/全量关系/画布尺寸；
  候选 README 增加正式入口与端点说明。
- `render/hf_reading_page.py`（新增）：正式阅读页——只内嵌 Archify 图、两阶段、
  四层导航、就近参数、完整文字、每条关系、源码/参数联查、未交付范围声明。

**新增行为**

- 整网层直接给出 Embedding 与输出端的算子级图（含权重/偏置就近节点）；
- 页面级 `?embed=1` 内嵌原生 Archify 图，并在每图后给出完整文字与逐条关系；
- 只有 Prefill/Decode 两个阶段，阶段切换只改变“本阶段约束”，不改符号；
- 页首/各层明确 D02–D04 未交付，不生成空白占位或假链接。

## 4. 验证结果（通过）

### 4.1 测试

- 17 个测试文件、**521 个测试全部通过**（`D01-verify/tests.log`）。
- 新增 `tests/test_deepseek_reading_page.py`（15 项）：端点算子/权重/形状、
  HyperHead 非单节点、`logits_to_keep`、候选无截断、两阶段、四层路由、
  原生内嵌、完整文字/关系、参数/源码联查、内部锚点、未交付声明、确定性。
- 既有 Archify/DeepSeek 语义/分层页/构建交付/其他四模型结构测试全部通过
  （共享模块必要保护回归）。

### 4.2 静态显示与契约（`D01-verify/display-audit.*`）

- 29 张图、553 节点、712 条关系；D01 图主图截断 **0**；页面完整文字 `…` **0**；
- 页面路由 595、锚点 1901、相对文件 168：失效/缺失 **0**；外链 **0**；
- 33 个 iframe 全部 `?embed=1`；无旧 SVG；无旧数值场景；
- 投影主字号估算 10.93–17.62px（network 最小，仍高于 6px 下限）。

### 4.3 原生 Archify 回执（`D01-verify/native-summary.md`）

- 29/29 图 `validate` / `deliver` / `check` 通过；
- 29/29 从候选 JSON 重建的 HTML 与交付 HTML **逐字节一致**；
- 29/29 delivery sidecar 的 `specification.sha256` 绑定候选 JSON、
  `artifact.sha256` 绑定交付 HTML。

### 4.4 历史保护

- 22242 个历史产物：**22242 未变 / 0 缺失 / 0 变更**；
- `CONTEXT.md`、`2026_10_17_workspace_structure_review.md` 保持删除；
- 新候选 Excel 与上一候选逐字节一致（`2dabd3af…365b05e`）。

## 5. 失败 / 跳过 / 不可用 / 未目检

| 类别 | 内容 |
| --- | --- |
| **不可用** | 浏览器门禁：29/29 图的 `browser-check` 在 Chrome DevTools 阶段失败（`Failed to initialize sandbox. Operation not permitted`）；正式页 `browser-check` 同样 exit 1。按交接要求未关闭 sandbox、未绕过限制，原始错误保存在 `D01-verify/native/<slug>/browser-check.json` 与 `native-evidence.json`。因此 AC18/AC19 **未执行**。 |
| **未目检** | 未保存任何截图，未人工逐张查看原生/嵌入页面；不宣称视觉通过。记录见 `D01-verify/visual-review.md`。 |
| **跳过** | 无测试被 `skipUnless` 跳过（本机只读 HF 输入齐全）。 |
| **部分** | 复用子图主图仍有 243 处既有省略号（页面完整文字面板为 0）；这些子图的排版修复属于 D02–D04/D05，本票未伪装已修。 |
| **失败** | 无最终构建/测试/图生成失败。构建过程中一次“编辑与运行竞争”的会话内候选已隔离，不计入证据（`handoff-check.md`）。 |

## 6. 已知限制

1. **浏览器与目检缺失**：本轮无法验证实际字体、iframe 内滚动/缩放、动态折叠、
   1440×900 与 1920×1080 下的嵌入比例和裁切；待环境允许后按 AC18/AC19 补做。
2. **内嵌高度策略**：iframe 高度按候选画布与页面最大内容宽度等比计算，宁留白不裁切；
   窄视口下会有额外留白，实际观感未目检。
3. **复用子图**：Decoder/mHC/Attention/KV/MoE 的排版、标签距离与省略号沿用现有生成器；
   本票只保证页面完整文字、全量关系与静态链接/几何校验，不宣称这些图的 SVG 文字已完整。
4. **端点粒度**：Embedding/输出端使用整网同源算子粒度（HyperHead 4 个具名算子 +
   权重/输入节点）；没有把 Norm/RoPE 等微操作进一步拆成设备任务，也没有新增源码中
   不存在的算子。
5. **阶段语义**：页面只切换“本阶段约束”（Prefill kvcachelen=0、Decode 符号），
   图形本身阶段无关；Excel 数值场景仍保留在导出物中，本轮阅读 UI 不展示。
6. **候选 `levels` 元数据**：`run_manifest.structure.archify.levels` 仍列 7 个层级名，
   端点图以 `entries`（29）与 README/页面入口体现；未另造层级名。

## 7. 停止点

D01 实施、测试与证据已完成，**停止等待 Codex 复核**：

- 未关闭 #17 或 #16，未修改 GitHub 标签/评论；
- 未执行 D02–D05，未继续旧 T14；
- 未 commit / push / reset / clean；
- 未恢复自动化，未归档聊天；
- 若需要浏览器/目检门禁，请在允许 Chrome sandbox 的环境重跑
  `D01-verify/collect_evidence.py` 与 `native_evidence.py`，并按 AC18/AC19 补截图。
