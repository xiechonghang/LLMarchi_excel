# T13 结果：Archify 接入 DeepSeek Decoder 产品分层阅读

状态：**T13 round-3（FFN 维度/全称收尾）完成，停止等待 Codex 复核**。未接 T14、未 commit/push、未关闭
[#14](https://github.com/xiechonghang/LLMarchi_excel/issues/14)、未改协调文件
（`execution.json`/`tickets.json`/`monitor-state.json`/任务 `README.md`/`executions/`）、
未恢复自动化（保持 PAUSED/45 分钟）、未归档聊天。T12 三轮证据、旧候选、五模型已验收
Excel 与既有删除原样保留。

最终候选：
`structure_reader/outputs/deepseek_v4_flash_0731_hf/19735c597e42a48da31384fc17cf8821757a81d49ddb0b2bd49c6f94bbab1393`
（analysis_id 同目录名）。入口
[`viewer/structure.html`](../../../structure_reader/outputs/deepseek_v4_flash_0731_hf/19735c597e42a48da31384fc17cf8821757a81d49ddb0b2bd49c6f94bbab1393/viewer/structure.html)：
整网 `#/network` → 任一层组 `#/group/<gid>` → `#/group/<gid>/block` →
Attention/KV、FFN/MoE、mHC 子图 → 逐级返回。证据目录
[`T13-decoder-verify/`](T13-decoder-verify/README.md)（本轮回修证据在
[`round-3/`](T13-decoder-verify/round-3/README.md)）。

## 1. 接手基线（逐文件哈希）

- cwd `/Users/xiechonghang/Workspace/LLMarchi_excel`，HEAD `947ae78f120279403ec0a4c539e99d3d8869e1d6`
  与交接一致（HEAD 不是实现身份：本票实现未提交，身份用文件/产物哈希表达）。
- 快照 `.../llmarchi-T13-decoder-vnext-baseline-20261030-fy_4uifk/manifest.json`，
  1158 文件 = 1153 逐字节一致 + 5 个快照后 Codex 协调更新（T13 交接单、任务 README、
  execution/tickets/monitor-state，只读登记）；14814 个受保护历史产物 14814/14814 一致；
  五模型 40 个当前候选文件 40/40 一致；既有两处删除仍缺失；协调目录外无未说明差异。
- 实现完成后再跑同一脚本：1148 一致 + 5 协调 + 5 个本票实现改动（`build_hf_decoder.py`、
  `cli.py`、`extract/deepseek_v4_semantic.py`、`archify_preview.py`、`hf_structure_page.py`）
  + 2 个本票新增文件，仍未说明差异 0；受保护产物 14814/14814、五模型候选 40/40 不变。
- 脚本/结果：[`verify_baseline_t13.py`](T13-decoder-verify/verify_baseline_t13.py) ·
  [`baseline-check.json`](T13-decoder-verify/baseline-check.json) ·
  [`baseline-check.md`](T13-decoder-verify/baseline-check.md)。
- T12 验收指定的 `T12-decoder-verify/round-3/` 与 `T12-decoder-review/` 均在上表保护集合内，
  未被覆盖或重生成；本轮以 round-3 语义（R1/R3/R5/R7 应用、R2/R4/R6 拒绝、mHC post/comb、
  缓存前态/后态）为唯一语义输入。

## 2. 复用 / 扩展 / 新增接口

**复用（不另造第二套结构事实或产品）**

| 接口 | 用途 |
| --- | --- |
| `extract/deepseek_v4_semantic.build_all_layer_group_bundles` | T12 同源语义图（四层组各一份） |
| `semantic/rules.normalize` + `validate_normalization` | T12 保守化简与审计（Archify 图直接用其结果） |
| `render/archify_preview.build_view_model` / `build_candidate` | Attention/KV 两层继续用 T12 视图导出，不重写 |
| `render/archify_preview.discover_archify` / `tool_identity` / `classify_browser_failure` | 工具发现、身份、浏览器失败窄分类（T12 既有能力） |
| `render/diagram.module_groups` | Block 模块分组（与产品 Block 总览同一份分组规则） |
| `render/display_names.shape_text` | 张量维度展开为 config/modeling 全称 |
| `render/hf_structure_page.render_structure_page` | 既有四层路由/面包屑/返回/算子卡，仅新增可选 `archify` 参数 |
| `build_hf_decoder._run_variants` + `_finalize_documents` | 既有分层构建、身份、原子落盘与幂等复用 |
| `cli build-hf` | 既有 CLI；新增 `--archify-layers` 开关 |
| `data/network_overview.json`、`data/layer_structures.json` | 图数据来源就是产品候选自己的两份事实文档（`path.operators`、`block.operators/tensors/source_refs`） |

**扩展**

- `render/archify_preview.finalize_candidate(...)`：把 T12 私有 `_run_finalize` 公开给产品层复用，
  并支持 `quality` 参数；`generate_previews` 行为不变（T12 证据脚本仍引用它，未删除）。
- `render/hf_structure_page`：`render_structure_page(..., archify=...)` 与 `_archify_panel` 卡片
  （图入口、门禁状态、candidate/完整关系 JSON、节点关系表、返回链接）。
- `build_hf_decoder.run_hf_build(..., archify_layers=False)`；`_run_variants` 内
  `_build_archify_layers` + `_archify_level_requests`；identity/`run_manifest.structure.archify`/README
  全部进入候选身份与交付清单。

**新增**

- `render/archify_layers.py`：七层图的程序化投影与布局
  （`build_network_graph` / `build_group_graph` / `build_decoder_graph` /
  `build_moe_graph` / `build_mhc_graph` / `layout_level` / `build_level_candidate` /
  `generate_archify_layers` / `archify_identity`），以及稳定交付面
  （`stable_entry` / `stable_manifest`）与子图包装页生成。
- `extract/deepseek_v4_semantic.ffn_named_linear_steps`：FFN 具名 Linear 展开
  （R-FFN-1/2/3，带成立条件与拒绝原因；只读旧事实，不改 Excel 粒度）。
- `tests/test_archify_layers.py`（27 项）。
- 本证据目录 4 个复核脚本 + 报告。

**没有删除任何被替代代码**：T12 的 `generate_previews`、预览证据与 CLI 旧路径仍有调用方
（`build_t12_evidence_r3.py` 与 T12 证据复现），按“先查引用再处理”的原则保留；T13 没有
第二套产品页/第二套模型事实。

## 3. 七层阅读路径与图内容

每层都由同一批事实文档程序生成（不是手抄、不是独立预览链接集合）：

| 层级 | 图 | 来源 | 内容 |
| --- | --- | --- | --- |
| 整网 | 1 | `network_overview.json#path` | token→embedding→mHC 扩展→43 层循环→HyperHead 收拢→Final Norm→logits；cos/sin、mask 为短旁枝；`past_key_values` 前态/后态虚线 |
| 层组 | 4 | `network.groups` + `layer_structures` | 43 层→本组真实层索引→代表层→Block 模板（模块/算子计数）；其余层组只作同构旁注 |
| Block/Decoder | 4 | `block.operators` + `module_groups` + 拓扑阶段切分 | 20–23 个阶段节点、26–39 条关系：input_layernorm、Q/KV/压缩器/Indexer/mask/core/output、mHC pre/post/comb→残差、主 KV 缓存前态/后态；互相依赖的模块被切成“KV 拼接 → mask 构建 → 核心计算”等真实阶段 |
| Attention | 5 | T12 语义 bundle | 化简后 4 组 + csa-topk 原始；33–41 节点、32–46 关系 |
| KV | 5 | T12 语义 bundle | 化简后 4 组 + csa-topk 原始；7–8 节点 |
| FFN/MoE | 4 | `block.operators` + `ffn_named_linear_steps` 同源投影 | 12–13 个复合算子 + 8 个具名步骤：路由/分派/单专家模板（一份）/共享专家/汇合；共享专家 gate/up/down 三个具名 Linear，专家模板按 gate_up_proj/down_proj 参数身份；新中间张量（gate/up 投影输出、SiLU 输出）在语义投影里登记形状/名称/来源，全部走全称映射 |
| mHC | 4 | `block.operators`（mhc_* + res_*） | 22 个算子：flatten→project→split→pre/post/comb→collapse + 两段残差 expand/mix |

节点身份与来源：Linear 保留实例名（Attention/KV 图中 `kv_proj · Linear` 等）；
Block/mHC/MoE 图用真实算子名 + 模块尾名；维度全部用 `display_names` 全称
（`batch_size`/`seqlen`/`kvcachelen`/`hidden_size`/`head_dim`/`hc_mult`/`n_routed_experts`…）。
mHC 的 `post`/`comb` 在 Block 图里是真实进入残差的边，mHC 图里是独立算子与张量。

计数以候选内 `data/archify/manifest.json` 为准（本报告不把节点计数当验收）。

## 4. 图数据、来源与节点映射

- 每张图交付：`data/archify/<slug>.json`（candidate）、
  `data/archify/edges/<slug>-relations.json`（全部关系：张量 id、标签、全称维度、kind、
  label；模块对的合并关系在 `members[]` 里逐条列出成员张量）、
  `data/archify/receipts/<slug>/`（finalize/browser-check 回执）。
- 映射：[`node-map.json`](T13-decoder-verify/node-map.json) 覆盖 27 张图，逐图给出
  节点（候选组件 id、算子 id、模块路径、源码行）与关系（张量 id/全称维度）；
  `build_evidence.py` 用同一批构建器与同一批输入重放，不调用 Archify。
- 来源：`hflib/transformers` @ `536ecc007387a50e77603bb5d92100e9b07514cc`，组件来源为
  该仓库相对路径与行区间，`finalize --repo-root` 现场核验；权重不进入数据流（MoE 的
  `router_bias`/`expert_*` 等 `kind=weight` 张量不作为边）。
- 原始/化简对照：Attention/KV 以 `raw`/`normalized` 成对出现（csa-topk 两组；其余层组
  化简后），Archify 卡片里逐条列出 R1–R7 应用/拒绝与条件，拒绝项保留原结构。

## 5. 布局与可读性

- 主链按数据依赖的拓扑秩自上而下；分叉向右、同秩横向；互依模块（如 `attn_core` 与
  `attn_mask`）放进同秩强连通分量，用横向关系表达，不制造回流。
- 跨秩关系（mHC post/comb→残差、hidden_streams→残差等）走右侧竖井，横段只落在行间空隙；
  布局层硬约束：任何数据边的目标秩不得高于源秩，状态边不得向上。
- 缓存：Block 层只画主 KV 缓存模块的前态/后态（其余缓冲/overlap/条目数在 Attention 层
  逐条展开），前态在消费者上方、后态在生产者下方，均为虚线 state 边。
- 可读性按 Archify `browser-check` 的真实投影字体门限（≥6px@1440×900）调过画布宽度：
  过宽会被判 `viewer/projected-text-readability`，过窄会 `sublabel` 溢出；最终 27 张图
  27/27 通过。主图放不下的长标签按像素预算截断，完整文字保留在节点全称卡片、
  candidate JSON 与 `node-map.json`。

## 6. 导航、离线与返回

- 分层页仍是单文件离线 HTML：27 个 `archify/*.html` 与 54 个 `../data/archify/*` 链接全部
  指向候选内真实文件（`candidate-verify.json`），无外部 `src`/外链资源；`file://` 直接可开。
- 每层有 Archify 卡片：子图入口（**同标签**打开候选内的 `.view.html` 包装页，包装页顶部有
  「← 返回层组 <gid> · Block 层」等可见返回链接并内嵌真实图）、具体原始图另开新标签、
  candidate/完整关系 JSON、关系清单（前 24 条 + 全部条数）。卡片只写稳定信息（节点/关系数、
  同源说明），不把某次浏览器可用性写进交付字节。
- 层组层与 Block 层各有「↑ 返回整网 / ↑ 返回层组」；算子层保留原「↑ 返回 Block/整网」。
- 真实 Chrome 点击走查（**22 步，0 错误**，[`round-2/browser-navigation.json`](T13-decoder-verify/round-2/browser-navigation.json)）：
  整网→四组层组→Block→返回层组→返回整网；csa-topk Block 上**实际点击**打开
  Attention / KV / FFN·MoE / mHC 四个子图（包装页内嵌真实图，读到具名
  `q_a_proj/q_b_proj/kv_proj/o_b_proj/gate_proj/up_proj/down_proj/gate_up_proj` 等 Linear 与
  全称维度），每次点击包装页「返回」回到 Block 层原上下文；再进入算子层（18 个模块折叠、
  150 个算子卡、54 行 Block 参数表，含 `L_kv`、全称表达式）后返回。

## 7. 真实浏览器与视觉证据

- 构建时 27 张图 `finalize` 的 `validate`/`deliver`/`check`/`browser-check` 全部通过
  （真实 Chrome 渲染 + 几何/可读性检查，不是“环境不可用”记录）；这些运行结果**不进入
  交付字节**（见 §8 的稳定交付策略）。
- 交付后对最终候选的 27 张 HTML 又逐张重跑 `browser-check`：**27/27 pass**，
  回执与每张 HTML 的 sha256 绑定保存在
  [`browser-checks.json`](T13-decoder-verify/browser-checks.json) 与
  [`browser-receipts/`](T13-decoder-verify/browser-receipts)。工具身份 Archify 3.2.0 / Node v26.10.0。
  回执含时间戳，故不进入候选交付清单（否则同输入重建无法幂等复用）。
- 12 张代表图在 Chrome 中从产品候选目录真实加载并渲染（非 JSON 断言）：
  节点/连线计数见 [`browser-diagrams.json`](T13-decoder-verify/browser-diagrams.json)
  （如 Attention csa-topk 化简后 74 个渲染节点/68 条连线，mHC 44/44）。
- 截图 10 张（1440×900，另 1 张 2048×1320 长图），索引
  [`screenshots/index.json`](T13-decoder-verify/screenshots/index.json)：
  [整网面板](T13-decoder-verify/screenshots/panel-network.png) ·
  [层组面板](T13-decoder-verify/screenshots/panel-group-csa-topk.png) ·
  [Block 面板（含各子图入口与关系清单）](T13-decoder-verify/screenshots/panel-block-csa-topk.png) ·
  [Block 图](T13-decoder-verify/screenshots/diagram-decoder-csa-topk.png) ·
  [KV 化简后](T13-decoder-verify/screenshots/diagram-kv-csa-topk.png) ·
  [Attention 长图](T13-decoder-verify/screenshots/diagram-attention-csa-topk-tall.png) ·
  [FFN/MoE](T13-decoder-verify/screenshots/diagram-moe-csa-topk.png) ·
  [mHC](T13-decoder-verify/screenshots/diagram-mhc-csa-topk.png)。
  dsh 实际查看了 Block 面板、整网面板、Decoder、KV、mHC 截图：具名 Linear、全称维度、
  单向下行、mHC 三路与残差、cos/sin 短旁枝、虚线缓存前/后态均可辨；27 张图的浏览器检查
  是自动验收，不宣称人工看完所有长图。
- 环境说明：T12 会话里 Chrome 因沙箱无法启动（当时按环境分类记录）。本会话在最终复核阶段
  可用 Chrome（文件策略为 full access，未请求额外提权），因此本轮浏览器门禁是真实通过，
  不是沿用环境豁免。

## 8. 新候选身份与可复现性

- analysis_id `f0b30340…` 由既有 identity（输入/工具/结构）+ 新增 `archify` 段哈希而成：
  Archify schema/builder、CLI/package 闭包/Node 版本、27 张图的 candidate/html 哈希、
  `render/archify_layers.py`、`render/archify_preview.py`、`render/hf_structure_page.py`、
  `semantic/rules.py`、`semantic/bundle.py`、`extract/deepseek_v4_semantic.py` 的哈希。
  未提交实现以文件哈希表达，未把根 HEAD 冒充为已提交实现。
- 稳定交付与运行证据分离（返修项 S2）：交付面只含与模型/图相关的稳定字段——每张图的
  candidate/HTML/包装页/关系哈希、节点/关系数、来源与工具身份；门禁结果、Chrome 是否可用、
  回执路径与时间戳一律不进 `data/archify/manifest.json`、产品页或 `run_manifest.json`。
  `_build_archify_layers` 还做一次白名单重建（`stable_entry`/`stable_manifest`），即使上游
  返回运行字段也不会泄漏到交付字节。
- 同输入重建可复现：图 candidate 与 HTML 由确定性构建器生成，Archify 交付 HTML 逐字节稳定。
  实测第二次完整构建返回同一 analysis_id 且 `reused: true`（不覆盖、不产生 diff）。
  跨环境验证：Codex 的 S2 探针（同一 candidate/HTML/tool，仅切换浏览器可用性）返回
  `same_identity=true, same_delivered_files=true, same_product_panel=true`。
  图数据、关系、manifest、原始 HTML 与 27 个包装页进入交付清单
  （115 文件，逐文件哈希见 `run_manifest.json`）。
- 工具身份与最终 HTML/浏览器证据绑定同一轮产物：`candidate-verify.json` 复算 27 个 HTML
  哈希 = manifest 记录值；截图与浏览器走查都直接打开该候选文件。

## 9. 测试

- 新增 `tests/test_archify_layers.py` **32 项**（第一轮 15 项 + 第二轮 12 项 + 第三轮 5 项）：
  层图构建（network/group/decoder/moe/mhc）、模块阶段拆分与无环硬约束、外部输入逐条接入、
  mHC 主出口与辅助边界、FFN 三个具名 Linear/专家参数身份与拒绝原因、模块对合并保留成员张量、
  全称形状、候选来源为仓库相对路径、页面 panel/包装页/返回/关系表、首次构建 `.work`、
  跨浏览器环境稳定交付、单层 + `--archify-layers` 具名拒绝（API/CLI）、FFN 中间张量形状派生/
  全称展示/形状校验器（拒绝空 shape、未知须显式说明）。
- 定向回归 **395 项通过**（[`round-3/tests-final.json`](T13-decoder-verify/round-3/tests-final.json)）：
  14 套件 338 项（含新增 32 项）+ `tests/hf/test_operator_workbook.py` 57 项。
- 既有问题单列、未计为通过：`test_hf_decoder_diagram` 6 项失败（T12 已单列的未跟踪夹具旧标签/
  缩写；本票未改 `decoder_page.py`/`diagram.py`，baseline 显示二者与接手快照逐字节一致）；
  `test_hf_decoder_onnx` 因 onnx/onnxsim/onnxruntime 未安装为空跑（skip）。
- Codex 的独立探针在最新候选上复跑：`S1/S2/S3 = false/false/false`
  （[`round-3/codex-build-probes.json`](T13-decoder-verify/round-3/codex-build-probes.json)），
  `t13-spec-probe` 四组 `SCC=[]`、`missing_inputs=[]`、出口张量正确
  （[`round-3/codex-spec-probe.json`](T13-decoder-verify/round-3/codex-spec-probe.json)），
  对应 round-2 FFN 形状探针的独立探针 4 组全部通过
  （[`round-3/ffn-shapes-probe.json`](T13-decoder-verify/round-3/ffn-shapes-probe.json)）。
- 未跑昂贵全量（与 T12 一致）。

## 10. 历史产物与工作簿保护

- 14814 个受保护历史产物 14814/14814 逐字节一致（含 T12 三轮证据、旧候选、单算子模板）；
  五模型 40 个已验收候选文件（含 `tables/decoder_variants.xlsx`）40/40 一致。
- 第一轮候选 `f0b30340…`、第一轮证据（`T13-decoder-verify/` 根目录）与 Codex 复核目录
  原样保留；本轮回修只新增 `T13-decoder-verify/round-2/` 与新候选目录。
- CUBE/VEC、收费/汇总、Excel 公式与导出器未改动；协调文件与自动化未改；未 commit/push；
  未关闭 Issue；未派下票。

## 11. 限制与说明

- **模块对合并**：Block/Decoder 阶段图把同一阶段对之间的多个张量合并为一条关系（与产品
  Block 总览粒度一致），成员张量逐条在 `relations.json#members[]` 与
  [`round-2/node-map.json`](T13-decoder-verify/round-2/node-map.json) 中；逐张量粒度在
  Attention/KV 图与 MoE/mHC 算子图。
- **raw 范围**：Attention/KV 的 `raw` 只对 csa-topk 提供（与 T12 验收一致），其余层组只给
  化简后；原始图与化简图的规则对照在 Archify 卡片与关系 JSON。
- **运行期未知量**：MoE 实际命中专家、每专家 token 数、缓存条目数仍是符号/未知，不臆测；
  图中未把 `kvcachelen` 与本次 `seqlen` 混用。
- **视觉**：主图长标签按像素预算截断（完整文字在卡片/JSON）；27 张图做了自动浏览器检查，
  并人工查看了 Block 面板、整网面板、Decoder/KV/mHC/MoE 图与子图包装页截图，未宣称看完
  所有长图的每个角落。
- **包装页与原始图**：产品入口进入 `.view.html`（同源包装页，内嵌图 + 可见返回路径）；
  原始 Archify HTML 仍单独交付并保留新标签入口，两者哈希均进入候选身份。
- **Chrome 可运行性**：构建时的 `browser-check` 依赖会话环境；受限沙箱会按 T12 的窄分类
  记为环境不可用（构建不因此伪造通过），交付字节不受影响（稳定交付策略）；需要时在可用
  Chrome 环境重跑 `browser-check` / 本目录走查脚本。

## 12. 第一轮复核的 7 项返修（round-2）

完整对照与独立证据见 [`T13-decoder-verify/round-2/README.md`](T13-decoder-verify/round-2/README.md)。

| # | Codex 发现 | 修复 | 独立回归 |
| --- | --- | --- | --- |
| S1 P1 | 首次构建缺 `.work` 父目录 | 先 `mkdir(parents=True)` 再 `mkdtemp` | Codex S1 探针 `reproduced=false`；测试覆盖全新临时 project_root |
| S2 P2 | 同一身份在不同浏览器环境产出不同字节 | 交付面白名单（manifest/页面/run_manifest 不含门禁/浏览器/时间戳），`_build_archify_layers` 二次白名单重建；回执归证据目录 | Codex S2 探针三项 `same_*` 均 true；测试 `test_delivery_identity_is_environment_independent` |
| S3 P2 | 单层 + 显式 flag 被静默忽略 | API 抛 `HfBuildError`，CLI 用法错误退出码 2，绝不静默 | Codex S3 探针 `HfBuildError`；两条新测试 |
| P1-1 | 子图漏掉视图外生产者提供的真实输入 | `_ops_graph`/MoE 段对**每个**跨边界非权重张量建边（含 `ffn_mhc_collapsed`、`attn_block_output`、`mlp_output`） | spec 探针 `missing_inputs=[]`；两条新测试 |
| P1-2 | mHC 出口不是声明的层输出 | `o-out` 只接 `exit_tensor` 的真实生产者；中间外部输出建独立辅助边界节点 | spec 探针 mHC 出口 `['hidden_streams_out']`；新测试 |
| P1-3 | SCC 同秩掩盖 core↔mask 双向箭头 | 按算子拓扑序把模块切成连续阶段（core 拆 KV 拼接/核心，mask 拆建模/扩展），并加“图中不得有 >1 的 SCC”硬约束 | spec 探针四组 `SCC=[]`；两条新测试（含双向图被拒） |
| P1-4 | FFN 三个具名 Linear 被复合框吞并 | 新增同源语义投影 `ffn_named_linear_steps`（R-FFN-1/2/3，带成立条件与拒绝原因）：共享专家 gate/up/down 三个 `· Linear`（真实 `shared_experts.*.weight`、模块路径、前后形状），专家模板按 `gate_up_proj`/`down_proj` 参数身份；旧算子事实与 Excel 粒度不变 | 新测试三条；浏览器走查在 MoE 子图读到 4 个具名 Linear |

返修后浏览器走查新增“**点击 Attention/KV→FFN→返回**”完整路径（22 步，0 错误）：
从 Block 面板实际点击进入四个子图包装页，读出具名 Linear/全称维度，点击包装页返回链接
回到原层组 Block 上下文，并检查算子层参数/源码详情后再返回层组/整网。

## 13. 第二轮复核的唯一剩余项：FFN 维度与全称（round-3）

Codex 第二轮确认原七项已解决，仅剩一项 P1：新 FFN 投影的四组图各 3 条中间数据边
`shape` 为空（`sem:shared_gate_up.gate_proj/.up_proj`、`sem:shared_act_down.silu`），
且 5 个 `· Linear` 组件仍显示 `['Es','H']`、`['2Es','H']` 旧别名。本轮只修该维度与
直接相关一致性，并补齐独立回归：

| 修复 | 实现 | 证据 |
| --- | --- | --- |
| 中间张量形状/名称/来源 | `extract/deepseek_v4_semantic.NamedLinearStep.tensor_facts` + `ffn_tensor_facts`：gate/up 形状 = 输入前缀 + 权重输出维（权重按 `[out,in]` 解释），SiLU 保持输入形状；名称由表达式派生、来源取算子源码行；无法证明时带 `reason`，图上以 `condition="未知：<原因>"` 显式标注 | [`round-3/ffn-shapes-probe.json`](T13-decoder-verify/round-3/ffn-shapes-probe.json)：四组共 12 条 `sem:` 关系全部非空、全称形状；0 缺失 |
| 全称展示 | 权重与输入/输出形状统一走 `render.display_names.shape_text`；Linear sublabel 形如 `shared_experts.gate_proj.weight [moe_intermediate_size,hidden_size]`、`gate_up_proj[expert_idx=e] [2×moe_intermediate_size,hidden_size]` | 同上探针 `alias_linear_components` 四组均为 0；[`round-3/screenshots/diagram-moe-csa-topk.png`](T13-decoder-verify/round-3/screenshots/diagram-moe-csa-topk.png) 可见全称形状 |
| 校验器拒绝空 shape | `render/archify_layers.validate_graph_shapes` 在出图前拒绝“内部真实数据边无 shape 且未标明未知原因”；未知必须写明原因 | 新测试 `test_shape_validator_rejects_unknown_internal_edge` 等 5 项 |

回归与证据：

- 新测试 5 项（形状派生、未知原因、中间边全称形状、Linear 全称、校验器正反例），
  `test_archify_layers` 共 32 项；定向回归 **395 项通过**
  （[`round-3/tests-final.json`](T13-decoder-verify/round-3/tests-final.json)）。
- Codex 既有探针复跑仍全绿：`S1/S2/S3=false`，四组 `SCC=[]`、`missing_inputs=[]`、
  出口张量正确（[`round-3/`](T13-decoder-verify/round-3/README.md)）。
- 四层组逐一检查（swa-hash/csa-hash/hca-topk/csa-topk），43 层归属、循环/未知规模、
  来源与旧入口不变；旧 Excel/收费/汇总未改。
- 27 张图交付后重跑 browser-check 27/27 pass；真实点击走查 22 步 0 错误；
  11 张截图（含 MoE 图，可读到全称形状与具名 Linear）。
- 前两轮候选（`f0b30340…`、`6d6440fa…`）、`round-2/` 证据与 Codex 复核目录原样保留；
  本轮候选 `19735c59…`，同输入重建 `reused: true`。
- Codex 报告其 IAB 对 `file://` 本地产品页受 URL 安全策略限制，未亲自重走；本轮不声称
  Codex 已重走页面。产品页/子图的真实浏览器记录与回执在本证据目录（dsh 会话内完成），
  FFN 维度缺陷的判定面是静态数据（relations/candidate），已由独立探针直接核对。

## 14. 停止点

T13 返修轮到此停止：等待 Codex 复核。T14（其余四模型）未开始；自动化保持 PAUSED/45 分钟；
协调文件、Issue 状态与 git 历史保持原样；未 commit/push、未关票。
