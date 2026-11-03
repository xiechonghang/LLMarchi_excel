# T14 结果：四模型推广与五模型数据流图 vNext 交付

状态：**T14 round-3 返修完成，停止等待 Codex 复核**（第一轮见 0A、第二轮见 0B）。未关闭
[#15](https://github.com/xiechonghang/LLMarchi_excel/issues/15)、未 commit/push、
未改协调文件（`execution.json`/`tickets.json`/`monitor-state.json`/任务 `README.md`/
`executions/`）、未恢复自动化（保持 PAUSED/45 分钟）、未恢复旧交接会话。T13
最终候选、T12/T13 全部证据、五模型 T11 工作簿、旧单算子模板、既有删除与全部历史
候选原样保留。

## 0B. 第二轮复核返修回应（round-3）

Codex 第二轮 [round-3-findings.md](T14-decoder-review/round-3-findings.md) 与
[整体评审](../../reviews/2026-10-31-version-review.md) 的阻断/应修项逐项处理；证据在
[T14-decoder-verify/round-3/](T14-decoder-verify/round-3/README.md)，旧候选、T13 DeepSeek、
T11 Excel 与历史证据保留。

| 复核项 | 本轮处理 | 证据 |
| --- | --- | --- |
| [P1] 830 条 lane 箭头缺 label/shape | lane `via` 分支不再提前跳过，统一写 `edge.shape`/`label`；侧栏短分支保留短标签（全称 shape 在节点 sublabel/relations）；所有 `via` data connection 均有 label | `round-3/candidate-verify.json`：四新候选 lane via 234/281/306/92，`lane_missing_label=[]`；测试断言 `candidate["connections"]` |
| [P1] tool_identity 带绝对路径 | identity 改为 `discovery/version/cli_sha256/node/package{digest,file_count}`，删除 `cli` 与 `package.root/files`；`BUILDER_VERSION=t14-archify-layers/4`；候选 manifest/analysis_id 无 `/Users/`；新增“换 HOME/安装路径 identity 不变”回归 | `round-3/candidate-verify.json` `identity_portable=true`；`tests/test_archify_preview.py::test_identity_is_portable_across_home_and_install_path` |
| [P2] HiDream ViT 组循环数 36 | `build_group_graph_for` 的 `g-loop` 使用本组真实层数；ViT 组为 27 层 | `round-3/candidate-verify.json` `group_loop_ok=true`；`tests/test_archify_layers_models.py::test_group_loop_uses_real_group_layer_count` |
| [P2] 索引/README 命令缺 `--archify-layers` | 五模型 README 生成命令与索引命令补 `--archify-layers`；按文档命令重建复现同一 analysis_id 另见 `reproduce-command.json` | 候选 README、索引；`round-3/reproduce_command.py` |
| [P2] 缺图 ↔ Excel 粒度登记 | 生成 `graph-excel-map.json`：每图逐节点映射 worksheet 行或明确 `not-billed`，按 variant 精确限定层组 | `round-3/graph-excel-map.json`、`round-3/graph_excel_map.py` |
| [P2] 主图卡片仍用缩写 | “参数（全称）”卡片用 `full_name`；节点/来源映射和 notes 统一 `expand_display_text`；Qwen 缓存 notes 改全称 | 候选卡片扫描 `card_alias_hits=[]`；测试 `test_cards_use_full_names` |
| [P3] normalized 仅“全部拒绝” | 明确登记“四模型当前没有可执行化简能力，0 条应用是能力边界，不是逐条审计结论”；normalized 卡片增加能力边界说明 | 本报告/索引说明；候选卡片标题“化简能力与规则” |
| 正式 browser-check | 默认保护仍为环境失败（Chrome sandbox 在 DSH 内不能初始化）；新候选需按用户 Terminal 流程重跑 81 图；本轮补充 opt-out 运行只作诊断，不计作默认保护通过 | `round-3/browser-check-default.json`（默认失败原文）、`round-3/browser-check-official.json`（opt-out 补充运行） |

## 0A. 第一轮复核返修回应（round-2）

Codex 第一轮 [round-1-findings.md](T14-decoder-review/round-1-findings.md) 的 5 项
逐项处理；返修证据全部在 [T14-decoder-verify/round-2/](T14-decoder-verify/round-2/README.md)，
未覆盖第一轮独立复核目录。

| 复核项 | 本轮处理 | 证据 |
| --- | --- | --- |
| [P1] 四新候选源码身份未闭合 | 所有共用源码冻结后重建四个候选；候选 `archify_product_sources` 7 项 + `tool_sources` 全部与当前文件 SHA 相等；`BUILDER_VERSION=t14-archify-layers/3` | `round-2/candidate-verify.json` 的 `source_identity_ok=true`（4/4） |
| [P1] 共享状态消费者漏线 26 条 | `_segment_state_nodes` / decoder 状态节点改为“节点复用、每个真实消费者一条状态读入边”；review 的 all-inputs probe 复跑 0 缺失 | `round-2/candidate-verify.json` `missing_input_edges=[]`；`tests/test_archify_layers_models.py` 新增多消费者回归 |
| [P2] 维度缩写（GLM 6、Qwen 3、HiDream 88） | 补齐 HiDream 参数/派生轴全称（`total_sequence_length`、`q_projection_width`、`kv_projection_width`、`pixel_patch_input_width`、`num_vision_attention_heads` 等），`expand_display_text` 覆盖节点/边/关系；修正 K 键语义不误展开为专家数；Qwen 模块标题改 `[hc_count,hidden_size]` | `round-2/candidate-verify.json` `old_alias_hits=[]`（4/4）；截图与候选扫描 0 命中 |
| [P2] cos/sin/mask 汇入顶部 `o-in` | `_ops_graph` 把 boundary/mask 辅助输入拆成每个消费点的局部 aux 短旁枝；`o-in` 只保留上游激活；真实 tensor id/全称维度/来源保留 | `round-2/browser-walk/*.json`；`tests/test_archify_layers_models.py` 的 aux 局部节点/边回归 |
| 正式 Archify browser-check | 默认标准流程仍复现 `viewer/browser-check-runtime`（Chrome 自身 sandbox 无法在 DSH 内初始化），**该项未完成**；另用 Archify 官方 `ARCHIFY_CHROME_NO_SANDBOX=1`（会关闭 Chrome sandbox）做补充诊断，81/81 图 finalize 通过且 HTML 与候选哈希一致，但**不计作默认保护下的正式通过**，交由 Codex 判断环境是否可接受；未改工具/伪造回执 | `round-2/browser-check-default.json`（默认失败原文）、`round-2/browser-check-official.json`（补充 opt-out 运行） |
| 最终重建/索引 | 冻结源码后重建四个新候选；T13 DeepSeek 与全部 T11 Excel/旧候选保留；索引不把未验收新版写成已验收 | 本报告第 1 节、`structure_reader/docs/model_delivery_index.md` |

## 1. 当前五模型入口

| 模型 | 当前候选 | 入口 | 图数量 |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731（T13 已验收） | `19735c597e42a48da31384fc17cf8821757a81d49ddb0b2bd49c6f94bbab1393` | [structure.html](../../../structure_reader/outputs/deepseek_v4_flash_0731_hf/19735c597e42a48da31384fc17cf8821757a81d49ddb0b2bd49c6f94bbab1393/viewer/structure.html) | 27 |
| GLM-5.3 | `dca02412faae0836ba49e332f4811c8dd350e589bb3a1080c3d8377c8b9bac18` | [structure.html](../../../structure_reader/outputs/glm_5_3_hf/dca02412faae0836ba49e332f4811c8dd350e589bb3a1080c3d8377c8b9bac18/viewer/structure.html) | 21 |
| GLM-5.3-Flash | `2343b957ee701b95018ac722bdbd72c855ff7f8e463d5ae61e9be1f1debd6c4c` | [structure.html](../../../structure_reader/outputs/glm_5_3_flash_hf/2343b957ee701b95018ac722bdbd72c855ff7f8e463d5ae61e9be1f1debd6c4c/viewer/structure.html) | 23 |
| Qwen3.8-Flash-Next | `df7ba03098ebd5051d7050fb38a363681fc0d3b9cfc3632a35f33cee5bbd62de` | [structure.html](../../../structure_reader/outputs/qwen_3_8_flash_next_hf/df7ba03098ebd5051d7050fb38a363681fc0d3b9cfc3632a35f33cee5bbd62de/viewer/structure.html) | 24 |
| HiDream-O1-Image | `701ed95da6bf70afd00c425e7e448acebd20729c55d63ddc7fbb7770c1654295` | [structure.html](../../../structure_reader/outputs/hidream_o1_image_hf/701ed95da6bf70afd00c425e7e448acebd20729c55d63ddc7fbb7770c1654295/viewer/structure.html) | 13 |

所有入口离线可用：`viewer/archify/*.view.html` 内嵌同候选单文件 Archify HTML，
有可见返回路径；`data/archify/<slug>.json`、`data/archify/edges/<slug>-relations.json`
给出 candidate 与全部关系；HiDream 另有 `data/archify/sources/<slug>-sources.json`。

## 2. 接手基线与保护

- cwd `/Users/xiechonghang/Workspace/LLMarchi_excel`，HEAD
  `947ae78f120279403ec0a4c539e99d3d8869e1d6` 与交接一致；HEAD 不是实现身份，
  本票使用 baseline manifest + 文件/候选哈希。
- `execution.json` baseline：1344 个快照文件 = 1334 逐字节一致 + 5 个快照后 Codex
  协调更新（T14 交接单、任务 README、execution/tickets/monitor-state）+ 5 个本票
  实现文件变化 + 0 个未说明差异；15227 个受保护历史产物 15227/15227 一致；
  T11 五模型 40 个当前候选文件 40/40 一致；既有 2 处删除仍缺失。
- 证据：[baseline-check.md](T14-decoder-verify/round-3/baseline-check.md) ·
  [baseline-check.json](T14-decoder-verify/round-3/baseline-check.json) ·
  [verify_baseline.py](T14-decoder-verify/round-3/verify_baseline.py)。
- 快照后协调文件只读登记，未修改；无关非协调差异 0。

## 3. 复用 / 扩展 / 新增

**复用（同一套事实、同一套产品管线）**

| 接口 | 用途 |
| --- | --- |
| `render/hf_model_profile.PageProfile.module_groups` | 五模型既有模块分组（页面与 Archify 图共用） |
| `render/archify_layers` T13 图模型/布局/candidate/包装页/稳定身份/白名单 | 五模型共用；DeepSeek 路径保留 |
| `render/archify_preview.discover_archify/finalize_candidate/tool_identity` | Archify 发现、门禁、工具身份 |
| `build_hf_decoder._run_variants/_finalize_documents` | 分层构建、身份、原子落盘、幂等 |
| `extract/deepseek_v4_semantic` + `semantic/rules` | DeepSeek csa-topk Attention/KV raw 对照、其余层组 normalized 与规则审计（T12/T13） |
| 既有 rules/extract/checks/render/export | 各模型真实结构事实与工作簿；未复制第二套事实 |

**扩展**

- `render/archify_profiles.py`（新增）：模型图声明——显示名、源码仓库/检出根/前缀、
  整网辅助算子、层组/Block 文案、子图分段（模块 id、页面标题、raw 对、入口/出口）、
  HiDream 整网阶段声明与“不声明 Archify 仓库证据”标志。模型差异不落到另一套构建器。
- `render/archify_layers.py`：通用整网/层组/Block/子图构建器；全缓存前态/后态；
  辅助输入短旁枝；阶段路由/净空修复；按模型选 raw/normalized；7 条化简规则审计；
  `sources` JSON 与包装页相对链接修复；`BUILDER_VERSION=t14-archify-layers/2`。
- `render/hf_structure_page.py`：Block 面板支持模型声明子图（不再固定 DeepSeek
  四槽位）；来源映射卡片与链接。
- `build_hf_decoder.py`：`ARCHIFY_LAYER_MODELS` 五模型；按模型请求/仓库/身份；
  同进程五模型构建走同一函数；HiDream 不声明仓库证据。
- `cli.py --archify-layers` 帮助文本更新为五模型。

**新增**

- `tests/test_archify_layers_models.py`（13 项）：模型声明、分段、来源、形状、旧别名、
  跨模型隔离、HiDream 单次网络、页面子图、未知层级拒绝、包装页链接。
- 本证据目录 6 个脚本/报告（见第 8 节）。

没有删除被替代代码；T12 的 `generate_previews`、T13 的 DeepSeek 语义路径仍保留。

## 4. 逐模型：范围、入口、证据与限制

### 4.1 GLM-5.3（`glm_5_3_hf`）

- 入口：
  `structure_reader/outputs/glm_5_3_hf/dca02412…/viewer/structure.html`；
  整网 → 3 组（`full-dense` 0-2、`shared-moe` 57 层、`full-moe` 18 层）→ Block →
  Attention / DSA Indexer / 压缩 KV / FFN（dense 或 MoE）。
- 真实语义：Q 低秩支路 + 压缩 KV latent + full 层独立 DSA indexer；shared 层不实例化
  indexer，`prev_topk_indices` 作为边界输入；dense 0-2 与 TopK MoE 分开；main KV 与
  indexer 键缓存各自前态/后态。
- 证据：21 图；[candidate-verify.json](T14-decoder-verify/round-3/candidate-verify.json) 中
  `glm_5_3_hf`；[browser-walk/glm_5_3_hf.json](T14-decoder-verify/round-3/browser-walk/glm_5_3_hf.json)
  （28 步、14 子图、21 图渲染、0 链接缺失）；截图
  [network](T14-decoder-verify/round-3/screenshots/glm_5_3_hf-network.png)、
  [decoder](T14-decoder-verify/round-3/screenshots/glm_5_3_hf-decoder.png)、
  [attention](T14-decoder-verify/round-3/screenshots/glm_5_3_hf-attention.png)、
  [wrapper](T14-decoder-verify/round-3/screenshots/glm_5_3_hf-wrapper.png)；节点/来源在
  [node-map.json](T14-decoder-verify/round-3/node-map.json)。
- 限制：MTP 层 78 不在主干；无视觉路径；无 ONNX；shared 层 indexer 结果来自上一层，
  不在本层重算（图注已说明）；长 Attention 图需滚动。

### 4.2 GLM-5.3-Flash（`glm_5_3_flash_hf`）

- 入口：
  `structure_reader/outputs/glm_5_3_flash_hf/2343b957…/viewer/structure.html`；
  3 组 `kda-dense`(0-2) / `dsa-moe`(3,7…43) / `kda-moe`(31 层)；Block 子图
  Attention（KDA 或 DSA）、KV/跨调用状态、DSA Indexer、mHC、FFN/MoE。
- 真实语义：KDA conv state 固定 Kconv、recurrent state 固定 [B,NhL,DhL,DhL]；
  DSA 压缩 KV latent 与 packed indexer 缓存（k+gate+valid）；NoPE（Rk=0）；
  mHC 4 路流 attn/ffn 两站点坍缩/pre/post/comb/写回；专家模板单份。
- 证据：23 图；browser walk 30 步、16 子图；截图
  [network](T14-decoder-verify/round-3/screenshots/glm_5_3_flash_hf-network.png)、
  [decoder](T14-decoder-verify/round-3/screenshots/glm_5_3_flash_hf-decoder.png)、
  [attention](T14-decoder-verify/round-3/screenshots/glm_5_3_flash_hf-attention.png)、
  [wrapper](T14-decoder-verify/round-3/screenshots/glm_5_3_flash_hf-wrapper.png)。
- 限制：KDA 无 softmax 注意力矩阵，不套 KV cache；DSA 的 Rk=0 槽位按源码保留为
  0 维事实；无视觉/ONNX。

### 4.3 Qwen3.8-Flash-Next（`qwen_3_8_flash_next_hf`）

- 入口：
  `structure_reader/outputs/qwen_3_8_flash_next_hf/df7ba030…/viewer/structure.html`；
  3 组 `gdn-moe`(35 层) / `gdn-ple-moe`(layer 1) / `qsa-moe`(12 层)；Block 子图
  GDN 或 QSA Attention、KV/状态、QSA Indexer、N-gram PLE、Gated Residual、FFN/MoE。
- 真实语义：GDN conv/recurrent state；QSA K/V 缓存 + 逐 token indexer key +
  micro-block 池化/块 top-k/tail；PLE 仅 layer 1，上下文 [B,2] 与 9 宽短卷积 state；
  Gated Residual 读门 [4,H] + 写门 [4]；MoE softmax 512 选 10，共享专家 640 宽 +
  sigmoid 标量门。
- 证据：24 图；browser walk 31 步、17 子图；截图
  [network](T14-decoder-verify/round-3/screenshots/qwen_3_8_flash_next_hf-network.png)、
  [decoder](T14-decoder-verify/round-3/screenshots/qwen_3_8_flash_next_hf-decoder.png)、
  [attention](T14-decoder-verify/round-3/screenshots/qwen_3_8_flash_next_hf-attention.png)、
  [wrapper](T14-decoder-verify/round-3/screenshots/qwen_3_8_flash_next_hf-wrapper.png)。
- 限制：PLE 只在真实 layer 1 出现；GDN 无 softmax 矩阵；QSA 无完整块的 query 走
  tail 分支；无视觉/ONNX。

### 4.4 HiDream-O1-Image（`hidream_o1_image_hf`）

- 入口：
  `structure_reader/outputs/hidream_o1_image_hf/701ed95d…/viewer/structure.html`；
  整网 12 个真实阶段（文本/时间步/视觉塔+2D RoPE/视觉 merge/DeepStack/条件构造/
  patch 投影/拼接/RoPE+AR 索引/UiT 36 层/Final 投影/输出选择）；两层组
  `uit-decoder`(36) 与 `vit-tower`(27，Kref>0)；子图 Attention（两趟）/SwiGLU FFN/
  DeepStack；ViT 组对应视觉注意力/MLP/tap。
- 真实语义：单次 forward 文本 + 像素 patch + 时间步 → x_pred；无 lm_head、无
  KV cache；AR 行因果 + 生成行全序列双向，两趟合并；DeepStack 只在
  `visual_pos_masks` 条件行注入；外部 50/28 步采样、CFG、scheduler、像素反拼不在范围。
- 证据：13 图；browser walk 18 步、8 子图、13 图；截图
  [network](T14-decoder-verify/round-3/screenshots/hidream_o1_image_hf-network.png)、
  [decoder](T14-decoder-verify/round-3/screenshots/hidream_o1_image_hf-decoder.png)、
  [attention](T14-decoder-verify/round-3/screenshots/hidream_o1_image_hf-attention.png)、
  [wrapper](T14-decoder-verify/round-3/screenshots/hidream_o1_image_hf-wrapper.png)；
  来源映射 `data/archify/sources/` 与
  [node-map.json](T14-decoder-verify/round-3/node-map.json)。
- 限制：源码是嵌套官方检出（`models/HiDream/HiDream-O1-Image`），不是 git
  top-level，因此**未声明 Archify 仓库证据**（不冒用根 HEAD）；文件 SHA 与逐节点
  来源在 `reports/source_manifest.json` + `data/archify/sources/`。视觉塔只在
  Kref>0 执行；无 ONNX。

### 4.5 DeepSeek-V4-Flash-0731

- 当前仍指向 T13 已验收候选 `19735c59…`（27 图、四层组；仅 csa-topk 的
  Attention/KV 提供 T12 raw 对照，其余层组 normalized；MoE/mHC 沿用 T13），
  本票不替换其身份；同进程构建只用于五模型互不污染核对。
- 保护：T13 最终候选与 T12/T13 证据均在受保护集合内，15227/15227 一致。

## 5. 同进程五模型与跨模型隔离

- [cross-model-build.json](T14-decoder-verify/round-3/cross-model-build.json)：同一 Python
  进程按 `GLM → Qwen → DeepSeek → HiDream → Flash` 构建五模型图；
  GLM 85/85、Qwen 97/97、HiDream 66/66、Flash 93/93 个生成文件与各自最终候选
  逐字节一致；DeepSeek 生成 109 个文件（T13 旧 builder 身份，不比较字节），
  只核对 27 图与身份。
- 每个模型 identity.model_key/levels/repository 独立；GLM indexer 不出现在 Qwen，
  HiDream 嵌套来源不进入 HF 模型；未知键与“单层 + archify”组合均具名拒绝并记录。
- 五模型同一构建器/布局/白名单，没有为某个模型复制整套构建器。

## 6. 来源、身份与 Excel 保护

- GLM/Flash/Qwen/DeepSeek：`hflib/transformers` @
  `536ecc007387a50e77603bb5d92100e9b07514cc`，Archify `--repo-root` 现场核对
  节点 source_refs；HiDream：官方实现 `2c2d29ff729e48f33e41f49edfdbd81d5ac103b4`，
  嵌套检出，独立 sources JSON + source_manifest 哈希，不声明仓库证据。
- 候选身份进入 `analysis_id` 的文件：结构文档、工具源、模型规则/提取/渲染源码、
  Archify builder/levels/repository/tool identity、每张图 candidate/html/view/
  relations（HiDream 另有 sources）哈希。图表回执/浏览器可用性/时间戳不进入候选字节。
- 新候选 identity 已去绝对路径（`tool_identity` 仅 discovery/version/cli_sha256/node/package.digest）；
  T13 DeepSeek 历史候选 manifest 仍保留旧工具路径，属 T13 已验收历史身份，本票不强制重建，
  已在索引登记；round-3 四个新候选无 `/Users/`。
- 五模型 `tables/decoder_variants.xlsx` 与 T11 逐字节相同（SHA 见交付索引）；
  `data/layer_structures.json`、`data/network_overview.json` 与 T11 逐字节相同。
  图的新粒度/子图/具名 Linear 不改变 Excel 行、CUBE/VEC 公式、收费与汇总。
- 旧候选、旧单算子模板、T12/T13 证据、既有删除全部保留；未删除任何历史来掩盖差异。

## 7. 测试与浏览器证据

- [round-3/tests-final.json](T14-decoder-verify/round-3/tests-final.json)：本票/复用/回归
  267 项通过（32+23+22+15+25+61+17+27+30+15；ONNX 20 项 skipped 另计）。
  历史 `test_hf_decoder_diagram` 54 项中 6 项失败（T13 已登记的旧 diagram 期望），
  单列不计通过；ONNX 20 项因本机缺依赖 skipped，单列不计通过。
- 文档命令复现：[round-3/reproduce-command.json](T14-decoder-verify/round-3/reproduce-command.json)
  四新模型按 README/索引的 `--all-variants --archify-layers` 重建，analysis_id 4/4
  与最终候选相同；五个 README 与索引命令均含该 flag。
- [round-3/candidate-verify.json](T14-decoder-verify/round-3/candidate-verify.json)：
  四新候选 `source_identity_ok=true`、`state_consumers_ok=true`、
  `missing_input_edges=[]`、`old_alias_hits=[]`、`card_alias_hits=[]`、
  `empty_data_edge_shapes=[]`、`identity_portable=true`、`group_loop_ok=true`；
  lane via 标签缺失 0（234/281/306/92 条），每图节点/关系与构建器重放一致，
  包装页链接 0 缺失。
- 正式 Archify browser-check：**默认标准流程仍未通过**。默认环境 4 个代表图均报
  `viewer/browser-check-runtime`（Chrome sandbox 在 DSH 内初始化失败，原始诊断见
  [browser-check-default.json](T14-decoder-verify/round-3/browser-check-default.json)）；
  该门禁项未完成，按复核要求保留原始诊断、不冒充通过。
- 补充诊断：[round-3/browser-check-official.json](T14-decoder-verify/round-3/browser-check-official.json)
  使用 Archify 官方 `ARCHIFY_CHROME_NO_SANDBOX=1`（关闭 Chrome sandbox）重跑 81 张
  最终图的 `finalize`，validate/deliver/check/browser-check 81/81 通过，HTML SHA
  与候选 `html_sha256` 逐字节一致。该 opt-out 不视为默认保护下的正式通过，
  是否接受由 Codex 判断；未修改工具、未跳过检查。
- [round-3/browser-navigation.json](T14-decoder-verify/round-3/browser-navigation.json) 与
  [round-3/browser-walk/](T14-decoder-verify/round-3/browser-walk/)：真实 Chrome 五模型共
  107 步、55 个子图打开/返回、81 张原始图渲染、0 链接缺失；每模型实际点击
  Block → 子图包装页 → 返回，并打开算子卡/参数表。该走查是补充证据，不替代正式门禁。
- [round-3/screenshots/index.json](T14-decoder-verify/round-3/screenshots/index.json)：20 张代表
  渲染图（每模型 network/decoder/attention/wrapper）；已查看 GLM/Flash/Qwen
  的 Attention、Qwen/Flash 的 Block、HiDream 的整网阶段图与包装页。

## 8. 证据与复现（round-3）

- 基线：[round-3/verify_baseline.py](T14-decoder-verify/round-3/verify_baseline.py) →
  `baseline-check.json/.md`
- 候选：[round-3/verify_candidates.py](T14-decoder-verify/round-3/verify_candidates.py) →
  `candidate-verify.json`、`node-map.json`
- 正式门禁：[round-3/browser_check_official.py](T14-decoder-verify/round-3/browser_check_official.py) →
  `browser-check-default.json`、`browser-check-official.json`、`browser-finalize/`
- 同进程五模型：[round-3/cross_model_build.py](T14-decoder-verify/round-3/cross_model_build.py) →
  `cross-model-build.json`
- 测试：[round-3/run_tests.py](T14-decoder-verify/round-3/run_tests.py) → `tests-final.json`
- 浏览器：[round-3/browser_walk.py](T14-decoder-verify/round-3/browser_walk.py) →
  `browser-navigation.json`、`browser-walk/*.json`
- 截图：[round-3/screenshots.py](T14-decoder-verify/round-3/screenshots.py) →
  `screenshots/*.png`、`index.json`
- 图↔Excel：[round-3/graph_excel_map.py](T14-decoder-verify/round-3/graph_excel_map.py) →
  `graph-excel-map.json`
- 默认保护复跑脚本：[round-3/run-browser-check.sh](T14-decoder-verify/round-3/run-browser-check.sh)
  （普通 macOS Terminal；81 图 finalize，不使用 sandbox opt-out）
- 文档命令复现：[round-3/reproduce_command.py](T14-decoder-verify/round-3/reproduce_command.py) →
  `reproduce-command.json`
- 统一索引：[model_delivery_index.md](../../../structure_reader/docs/model_delivery_index.md)

## 9. 停止点

T14 已完成，停止等待 Codex 复核。不关闭 #15、不宣布验收通过、不 commit/push、
不改自动化/协调文件、不恢复旧会话；Codex 可据本报告与证据目录独立复算。
