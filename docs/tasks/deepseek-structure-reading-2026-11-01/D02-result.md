# D02 Decoder 与 mHC 完整分支数据流结果

日期：2026-11-02（Asia/Shanghai）。范围：D02/#18。未启动 D03–D05，未关 Issue，
未 commit/push，未恢复自动化，未访问/恢复旧 T14 会话。状态：**实施、候选与可运行
原生/静态验收完成；浏览器截图与目检待 Codex 授权环境补跑**。

## 1. 新候选

| 项 | 值 |
| --- | --- |
| 候选 | `structure_reader/outputs/deepseek_v4_flash_0731_hf/0eb1ca6461639b112c52c4bd7df0494ce2b971425307b6fdcc8d1d9e6ecfe1b7` |
| 正式入口 | `viewer/structure.html`（2,027,885 bytes，sha256 `3a86694d5771737058a3a5b98ef6e959512e8c76c0cd21e43b761b573045d70c`） |
| 整网原生图 | 保持 D01 验收字节：sha256 `42c48b1a7d74c2e7a69f5e8385625e2334c252ef92d7570cd72894a06ccaf75d` |
| 构建命令 | `PYTHONPATH=structure_reader/src <python> -c "from structure_reader.cli import main; import sys; sys.exit(main(['build-hf','--model','deepseek_v4_flash_0731_hf','--all-variants','--archify-layers','--json']))"` |
| 证据 | `D02-verify/` |

## 2. 源码依据与改动

结构事实来自既有 DeepSeek 静态结构管线（config + `DeepseekV4DecoderLayer/attn_hc/ffn_hc`
源码事实），不实例化模型、不读取真实权重。关键 config 事实包括
`hc_mult=4`、`hc_sinkhorn_iters=20`、`num_hidden_layers=43`；
mHC 两段来自 `hc_mult` 的 `(2+Nhc)×Nhc` 权重投影、split、sigmoid 门控、softmax+Sinkhorn
comb 与 broadcast-mul+sum collapse。复用既有 `build_*_graph`、Archify 生成器与
D01 的正式页面，没有另建手工图。

### Decoder（`render/archify_layers.py`）

按四个真实层组生成**分支级** Decoder：

- mHC 节点保留两段 `attn_hc` / `ffn_hc`，label 明确“返回 post / comb / collapsed”；
  decoder 中不把这些门控塞回 mHC 节点内部。
- collapsed → Norm → 既有 Attention 阶段；ffn 段 collapsed → Norm → 既有 MoE 阶段。
  Attention/MoE 仍是可进入的模块阶段节点，并标注“模块未展开：细节见对应页签”。
- Attention/MoE 的 `branch_output` → post 展开；post + branch_output 生成 expanded。
- comb → combᵀ 转置；combT + expanded + 原始多流 → combᵀ 混合 + Add；输出进入下一段。
- 两段结构相同但层组不同：swa-hash 24 节点/36 边；csa-hash 27/45；hca-topk 27/43；
  csa-topk 27/45。差异来自真实层型的 Attention/MoE 模块，而不是模板复制。
- 没有 mHC 直接连 Attention/MoE 节点的边；没有向上回流；状态缓存前后态仍由
  `_add_module_state_nodes` 保留。

### mHC 权重细图

`build_mhc_graph` 重写为**只含 mHC.forward 内部**的细图：

- 两段各 7 个具名算子：flatten/norm、project、split、pre、post、comb（含 Sinkhorn）、
  collapse；两段合计 14 个算子节点。
- pre 留在主链；post/comb 作为 split 的两个并列输出放在旁枝竖列，避免 rank 布局把
  4 个门控并排拉宽。
- 每段增加一个 `Decoder.forward` **消费边界**节点，显式接收 collapsed/post/comb；
  节点文字说明“collapsed→Norm→分支；post→展开；comb→combᵀ+Add”，且注明不是
  mHC 内部调用。
- 不画 Attention/FFN 调用、不画 o-out、不把残差展开/Add 放进 mHC。
- 图节点 17、边 22；参数、Sinkhorn（Sit=20）与来源仍在节点/参数表/来源清单中。

### 页面

- `viewer/structure.html` 的 banner 更新为 **本票（D02）交付**；D03–D04 的
  Q/KV/O/缓存与 Hash/TopK/专家内部仍明确“尚未交付”。
- 四个 decoder 图位、四个 mHC 图位仍由既有页签承载；Attention/MoE 未展开入口
  指向既有同源子图，不新增假链接。
- 保留 D01 两阶段符号、整网端点、页签独立、来源/权重依赖、全图库不省略文字、
  标签防重叠。

## 3. 需求对应

| 验收点 | 证据 |
| --- | --- |
| AC05 导航 | 页面 33 图位/9 页签组/22 按钮；decoder 4、mHC 4；路由/锚点失效 0 |
| AC07 抽象 | decoder Attention/MoE 节点带“模块未展开”说明；mHC 节点指向 mHC 页签；mHC 细图只到具名算子 |
| AC08 mHC | 两段 collapsed/post/comb、branch_output、post 展开、combᵀ、Add；无 mHC 内部调用 Attention/FFN |
| AC13 参数 | 候选参数表保留全称/来源；mHC comb/Sinkhorn 参数在同一候选参数表 |
| AC14 文字 | 29 图主图/连线省略号 0；页面完整文字行省略号 0 |
| AC15 连线 | Decoder/mHC 关系来自真实 tensor inputs/outputs；每个数据边有 shape；返回三路各一条 |
| AC16 几何 | `geometry-audit.json`：节点/via/标签外沿 min x/y ≥0，标签不压节点；mHC 左裁切专项通过 |
| AC17 标签 | 29 图显式标签重叠 0、SVG 可见文字重叠 0；33 图位短 iframe 0 |
| AC18/19 | **未完成**：Chrome 环境失败，0 截图，未目检；见 §5 |
| AC20 来源 | 新候选 HTML/页面探针均绑定 sha256；旧候选证据未复用 |
| AC21 保护 | 22242 + 248 + f1ea 124 全部未变；Excel 逐字节一致；两处删除保持 |
| AC22 诚实 | 浏览器/截图/目检单列未完成；未把离线几何写成视觉通过 |

## 4. 测试与原生

- **545 项测试全部通过**（17 文件，`D02-verify/tests.log`）；新增 5 项 D02 回归：
  Decoder 两段 mHC 返回/拆分、向下流动、mHC 细图返回与无分支节点、几何外沿、
  标签不压节点。
- 29/29 图 `validate`/`deliver`/`check` 通过；29/29 重建 HTML 与交付字节一致；
  delivery sidecar 绑定候选 JSON/HTML（`native-summary.md`）。
- 29 张唯一图：442 个显式 labelAt + 156 个 attention/kv 自动标签；显式标签重叠 0、
  SVG 可见文字重叠 0、省略号 0。
- 33 嵌入图位：viewBox 高度不足 0。
- `geometry-audit.json`：4 个代表 decoder/mHC 图的外沿 min x/y=20/70，标签不压节点，
  mHC 最大外沿右 1180 < viewBox 1221。

## 5. 浏览器限制

dsh 当前环境仍无法启动 Chrome：

- `Failed to initialize sandbox. Operation not permitted`
- `GPU process exited unexpectedly: exit_code=6`
- `Timeout Page.enable`

已实际运行并保留原始回执：

- `D02-verify/probe-attempt/`（`verify-final.mjs --geometry-only`，exit 1，0 截图）
- `D02-verify/probe-attempt-full/`（完整探针，exit 1，0 截图）
- `D02-verify/visual-attempt/decoder-swa-hash/`、`mhc-swa-hash/`（visual-check exit 1）
- `D02-verify/native-evidence.json`（browser-check 29/29 环境失败）

未关闭/绕过 Chrome sandbox，未修改探针断言。因此：

- 1440×900 / 1920×1080、100% 缩放、查看器缩放、长图滚动、跨层返回、参数/源码
  联查的最终候选实测未完成；
- 29 图逐张截图与目检 0 张；
- AC18/AC19 及 D02 浏览器视觉验收保持未完成，待 Codex 补跑。

## 6. 历史保护与协调

- 继承 22242 项历史产物：22242/22242 未变；
- 上轮候选 248：248/248 未变；
- D02 派发前 f1ea240e… 的 124 文件：124/124 未变；
- 两处既有删除保持；
- Excel 最新两候选逐字节一致（`2dabd3af…365b05e`）；
- 协调文件 1845/1847 未变；`tickets.json`、`D01-execution.json` 在快照后被
  Codex 外部更新为 D02 running（只读，未由本轮修改）；
- 源文件快照 202：198 未变，4 修改（archify_layers、hf_reading_page、两个测试）。

## 7. 限制

1. 浏览器/截图/目检未完成；文档未宣称视觉通过。
2. Attention/MoE 内部仍沿用既有子图，D03/D04 专门改进未实施；本票只保证 Decoder
   分支边界与入口。
3. mHC 图只覆盖 mHC.forward 内部；Decoder.forward 的残差/Add 在 Decoder 图。
4. 离线标签几何是字符宽度估算，最终 `getBBox`/截图以授权环境为准。

## 8. 停止点

完成 D02，停待 Codex 复核。未关闭 #18/#16，未启动 D03–D05，未 commit/push，
未恢复自动化，未修改 Codex 协调 JSON、既有验收证据或 handoff。
