# 五模型结构交付索引（T11 Excel 基线 + T14 vNext，2026-10-31）
## T14 当前 vNext 交付（2026-10-31，返修中；图入口暂以此为准）

T14 在 T12/T13 已验收的语义/化简/Archify 产品管线上推广到四模型，并统一五模型
当前入口。下表是**当前可查看的 vNext 图入口**；下文 T11 章节继续作为 Excel/旧表
基线与历史机器可读索引保留，不因 T14 改写。

| 模型 | 模型键 | 当前候选（analysis_id / 入口） | 图数量 / 层级 | 真实范围与未知量 |
| --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | `deepseek_v4_flash_0731_hf` | `19735c597e42a48da31384fc17cf8821757a81d49ddb0b2bd49c6f94bbab1393` · [入口](../outputs/deepseek_v4_flash_0731_hf/19735c597e42a48da31384fc17cf8821757a81d49ddb0b2bd49c6f94bbab1393/viewer/structure.html) | 27：network / group / decoder / kv / attention / moe / mhc | 43 层、4 组（`swa-hash`、`csa-hash`、`hca-topk`、`csa-topk`）；mHC 4 流；KV 前态/后态；T13 已验收身份 |
| GLM-5.3 | `glm_5_3_hf` | `dca02412faae0836ba49e332f4811c8dd350e589bb3a1080c3d8377c8b9bac18` · [入口](../outputs/glm_5_3_hf/dca02412faae0836ba49e332f4811c8dd350e589bb3a1080c3d8377c8b9bac18/viewer/structure.html) | 21：network / group / decoder / attention / indexer / kv / ffn | 78 层主干、3 组；full 层独立 DSA indexer、shared 层复用上一 full 层 top-k 索引；dense 0-2；MTP 层 78 不在主干；无视觉路径；无 ONNX |
| GLM-5.3-Flash | `glm_5_3_flash_hf` | `2343b957ee701b95018ac722bdbd72c855ff7f8e463d5ae61e9be1f1debd6c4c` · [入口](../outputs/glm_5_3_flash_hf/2343b957ee701b95018ac722bdbd72c855ff7f8e463d5ae61e9be1f1debd6c4c/viewer/structure.html) | 23：network / group / decoder / attention / indexer / kv / mhc / ffn | 45 层、3 组；KDA（34 层，conv/recurrent 固定状态）与 MLA/DSA（11 层，压缩 KV + packed indexer 缓存）；NoPE（Rk=0）；mHC 4 流；无视觉/ONNX |
| Qwen3.8-Flash-Next | `qwen_3_8_flash_next_hf` | `df7ba03098ebd5051d7050fb38a363681fc0d3b9cfc3632a35f33cee5bbd62de` · [入口](../outputs/qwen_3_8_flash_next_hf/df7ba03098ebd5051d7050fb38a363681fc0d3b9cfc3632a35f33cee5bbd62de/viewer/structure.html) | 24：network / group / decoder / attention / indexer / kv / ple / mhc / ffn | 48 层、3 组；GDN 36 层（conv/recurrent 状态）、PLE 仅 layer 1、QSA 12 层；Gated Residual 4 路读写门；MoE 512 选 10；无视觉/ONNX |
| HiDream-O1-Image | `hidream_o1_image_hf` | `701ed95da6bf70afd00c425e7e448acebd20729c55d63ddc7fbb7770c1654295` · [入口](../outputs/hidream_o1_image_hf/701ed95da6bf70afd00c425e7e448acebd20729c55d63ddc7fbb7770c1654295/viewer/structure.html) | 13：network / group / decoder / attention / ffn / deepstack | 单次 forward：UiT 36 层 + 条件 ViT 27 blocks（Kref>0）；文本 + 像素 patch + 时间步 → x_pred；无 KV cache；外部采样/CFG/scheduler/像素反拼不在范围 |

### 图内容与来源

- 每模型入口都是 `viewer/structure.html#/network` → `#/group/<gid>` → `#/group/<gid>/block`
  → 子图包装页（`viewer/archify/*.view.html`）→ 返回原 Block；层组真实索引、代表层和
  模板复用不代表共享权重。
- 子图按模型真实语义声明：GLM 的 full/shared indexer、Flash 的 KDA/DSA + mHC、
  Qwen 的 GDN/QSA/PLE/Gated Residual、HiDream 的两趟注意力/SwiGLU/DeepStack；
  不套用 DeepSeek 的缓存、mHC 或 logits 主路径。
- 维度使用 config/modeling 全称（`batch_size`、`seqlen`、`kvcachelen`、
  `hidden_size`、`head_dim`、`kv_lora_rank`、`hc_mult` 等），不显示 `B/S/H/Es`
  旧别名；每条关系在 `data/archify/edges/` 中带 tensor id、全称 shape、分支/状态。
- 缓存/循环状态一律“调用前状态 → 计算 → 调用后状态”单向表达，不用回流箭头；
  DeepSeek 主 KV 与 Flash/Qwen 的 conv/recurrent/KV/indexer/PLE 状态均分开。
- normalized 能力边界：四个新模型当前**没有可执行的化简变换**，raw/normalized
  结构相同；0 条应用是能力边界，不是逐条评估后的拒绝结论。normalized 卡片逐条列出
  7 类候选规则与“未证明/不适用”原因。
- 来源：HF 模型由 Archify `--repo-root hflib/transformers` 在
  `536ecc007387a50e77603bb5d92100e9b07514cc` 上核对；HiDream 官方实现位于嵌套检出
  `models/HiDream/HiDream-O1-Image`（`2c2d29ff729e48f33e41f49edfdbd81d5ac103b4`），
  该目录不是 git top-level，因此**不声明 Archify 仓库证据、不冒用工作区根 HEAD**：
  节点→路径/行号在 `data/archify/sources/`，文件 SHA256 在
  `reports/source_manifest.json`。

### raw / normalized / 化简规则

- DeepSeek 沿用 T13：Attention/KV 的 T12 `raw`/`normalized` 成对，T12 规则
  R1/R3/R5/R7 应用、R2/R4/R6 拒绝，理由在 T12/T13 证据与候选卡片。
- GLM/Flash/Qwen/HiDream 的 Attention 成对交付 `raw` 与 `normalized`；
  normalized 图卡片逐条列出 7 类候选化简及**拒绝原因**（本轮实际应用 0 条）。
  不做没有证明的 View/Transpose/Split/Concat 合并；原算子、张量和来源保留。

### 与旧 Excel/性能表的关系

- 五模型 `tables/decoder_variants.xlsx` 与 T11 已验收工作簿**逐字节相同**：
  - DeepSeek `2dabd3af336e08b59cc847b7cc8d5c89fc8bc46244abf3a685a0077cc365b05e`
  - GLM-5.3 `2aeeed05be4954e8ff31b32885b1dd44c991e228e9dce56c8df07f108e464647`
  - GLM-5.3-Flash `fe648a311a70e8e06b22220700c3522862b77f4b0f47172dfec19bd909f28d78`
  - Qwen3.8-Flash-Next `500d2685904c2cf500ea15c54e86236cbe98462bca2afe34cc8dfd2bff404813`
  - HiDream-O1-Image `9a028e7a207d9dd73b863fbf64a22460a7e19b5880ce108634cb95d062bd159e`
- 图的新粒度（具名 Linear、单独缓存前/后态、Indexer/PLE/DeepStack 子图）**不改变
  Excel 行、CUBE/VEC 公式、收费与汇总**；旧单算子模板与历史候选保留，不新增计费。
- MoE 专家模板按真实融合参数 `gate_up_proj`/`down_proj` 展示，不虚构独立
  nn.Linear 实例；共享专家/稠密 FFN 保留 `gate_proj`/`up_proj`/`down_proj` 实例名。
- **图 ↔ Excel 粒度登记**：[round-3/graph-excel-map.json](../../docs/tasks/model-structure-2026-10-28/T14-decoder-verify/round-3/graph-excel-map.json)
  逐图/逐节点列出对应 `Prefill/Decode` worksheet 行（按 variant 精确限定层组）或
  `not-billed` 原因。当前登记：GLM 330 mapped/149 not-billed、Flash 427/119、
  Qwen 446/146、HiDream 120/66、DeepSeek 389/117；图只复用 T11 既有算子行，
  **不新增行/计费项**。

### 验证与已知限制

- 证据目录：[T14-decoder-verify/round-3](../../docs/tasks/model-structure-2026-10-28/T14-decoder-verify/round-3/README.md)
  （第二轮复核返修：lane 标签、可移植身份、ViT 层数、构建命令复现、图↔Excel 映射、
  测试/浏览器走查/截图）。
- 历史 diagram 6 项失败（T13 已登记）与 ONNX 缺依赖单列，不计入通过；本票未改
  Excel 内部计算。
- 正式 Archify browser-check：DSH 沙箱内仍报 `viewer/browser-check-runtime`
  （Chrome sandbox 无法初始化），新候选的默认保护复跑需用户在本机 Terminal 执行；
  round-3 的 `ARCHIFY_CHROME_NO_SANDBOX=1` 补充诊断 81/81 图通过且 HTML 哈希一致，
  但**不计作默认保护通过**；默认保护复跑脚本：
  [`round-3/run-browser-check.sh`](../../docs/tasks/model-structure-2026-10-28/T14-decoder-verify/round-3/run-browser-check.sh)。第二轮旧候选的用户侧 81/81 记录见
  [`T14-decoder-review/round-2/browser-check-evidence.md`](../../docs/tasks/model-structure-2026-10-28/T14-decoder-review/round-2/browser-check-evidence.md)，
  不自动适用于新 analysis_id。
- #15 仍开放；2026-10-31 第三轮返修（lane 标签、可移植身份、ViT 层数、构建命令、
  图↔Excel 登记、卡片全称）见
  [`round-3-findings.md`](../../docs/tasks/model-structure-2026-10-28/T14-decoder-review/round-3-findings.md)
  与 [round-3 证据](../../docs/tasks/model-structure-2026-10-28/T14-decoder-verify/round-3/README.md)，
  待 Codex 复核。
- 新候选工具 identity 已去绝对路径；T13 DeepSeek 历史候选仍含旧 manifest 工具路径，
  按 T13 已验收身份保留、不重建，登记为历史差异。
- 长图仍需沿阅读方向滚动；主画布会截断长文字，完整节点/来源/关系在 candidate、
  relations 与 sources JSON。


> 本索引只记录**已落盘**的当前候选与统一入口。候选身份是对「输入官方文件 + 本工程工具源码」的内容寻址
> （`analysis_id`），不是 HEAD 源码身份；当前工作树未提交（HEAD `947ae78f120279403ec0a4c539e99d3d8869e1d6`），
> 不能把本页、候选或机器可读索引当作 HEAD 对应实现或用户阅读验收的证明。
>
> **T11 性能阶段（2026-10-29）**：五模型都已按用户确认的 CUBE/VEC 规则接入 Excel（逐算子分类、
> B/M/K/N 或 M/N、工作量、用户规则耗时、层组小计与整网串行汇总）。因共享导出器与各模型性能规则
> 源码变化，五个 `analysis_id` 全部更新；T05–T08 与 T09/T10 的历史候选原样保留、不作为当前。
> 与 T10 已验收的 DeepSeek 回归分两层：**成本公式/数值不变**（279 个默认场景成本单元格逐项相同、
> T10 手算锚点复现），结构 JSON 逐字节不变；但共享导出器的“先判条件再算维度”改动也改了 DeepSeek
> 条件算子 `hca_block_bias` 的维度/工作量/状态公式（Prefill/Decode 各 V/X/Y/Z 四格）：有效场景成本相同，
> false 分支维度显示 0。除此之外只有成本列标题（每算子区 1 处）与冻结窗格 B2→C2 变化。
> 详见 `T11-cube-vec-verify/regression.json`（按类别列出逐单元格差异）。

## 统一入口

```sh
cd /Users/xiechonghang/Workspace/LLMarchi_excel
PYTHONPATH=structure_reader/src \
  /Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  structure_reader/scripts/read_model_structure.py \
  build-hf --model <model_key> --all-variants --archify-layers --json
```

- 本机实测解释器 Python 3.12.14；核心仅标准库（读源码/写 HTML/写 xlsx 之外不装依赖）。
- 五模型**同一命令、同一种产物布局**：`structure_reader/outputs/<model_key>/<analysis_id>/`；
  stdout JSON 的 `output_dir` 就是入口目录，`analysis_id` 是内容寻址身份。
- 同输入 + 同工具源码再次构建会幂等复用（`reused: true`）；工具源码一变就是新候选，旧候选不覆盖。
- 每个模型只使用自己的来源/规则/提取/检查/导出模块；不跨模型借用，未知模型键具名拒绝。
- 显式 `--onnx` 在未实现该能力的四个模型上具名拒绝；缺少 `--all-variants` 时四模型也具名拒绝。
- 每页性能配置区有可改的 CUBE/VEC 吞吐（默认 4096 MAC/ms、64 元素/ms）；未知/非法输入显示“未知（原因）”，
  不按 0 计；整网总计只在全部贡献项已定时才是数值。

## T11 Excel/旧表当前候选（历史基线，不是 T14 图入口）

| 模型 | `model_key` | `analysis_id` | 覆盖 | 工作表 |
| --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | `deepseek_v4_flash_0731_hf` | `c4a2f98acd52a8f40b08481a84771c58f276957c6bbd9acf2da3900d24e9c4a5` | 43 层 / 4 组：`swa-hash`(0,1)、`csa-hash`(2)、`hca-topk`(3,5…41)、`csa-topk`(4,6…42) | Prefill/Decode |
| GLM-5.3 | `glm_5_3_hf` | `1ff1a8282b8b8675352231ca4bc499db9e20ce371362d634a8d12427a0644029` | 78 层 / 3 组：`full-dense`(0-2)、`shared-moe`(57 层)、`full-moe`(18 层) | Prefill/Decode |
| GLM-5.3-Flash | `glm_5_3_flash_hf` | `e0a72bc340d038c20926ceae09ed648bfad69bf20eed512eecd475db1b0a0cc8` | 45 层 / 3 组：`kda-dense`(0-2)、`dsa-moe`(3,7…43，11 层)、`kda-moe`(31 层) | Prefill/Decode |
| Qwen3.8-Flash-Next | `qwen_3_8_flash_next_hf` | `7b5bee0e46a04d56bffe1a79dbf1d0d08bdb47271d35b0204b70feafa8e8c17b` | 48 层 / 3 组：`gdn-moe`(35 层)、`gdn-ple-moe`(layer 1)、`qsa-moe`(3,7…47，12 层) | Prefill/Decode |
| HiDream-O1-Image | `hidream_o1_image_hf` | `b4cc406ace15beb62637cad18aa2b6f530e98d129cd969b32400ea23831d34de` | 单次 forward：`uit-decoder` 36 层 + `vit-tower` 27 blocks（Kref>0 条件执行） | Generation |

## 能力矩阵（显式支持/拒绝）

| 模型 | `--all-variants` 分层入口 | 单层 `--layer-index 4` | `--onnx` | 来源漂移/未知模型键 |
| --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | 支持 | 支持（兼容入口保留） | 能力入口保留；本票未重建（缺依赖） | 具名拒绝 |
| GLM-5.3 | 支持 | 不支持（显式拒绝） | 不支持（显式拒绝） | 具名拒绝 |
| GLM-5.3-Flash | 支持 | 不支持（显式拒绝） | 不支持（显式拒绝） | 具名拒绝 |
| Qwen3.8-Flash-Next | 支持 | 不支持（显式拒绝） | 不支持（显式拒绝） | 具名拒绝 |
| HiDream-O1-Image | 支持 | 不支持（显式拒绝） | 不支持（显式拒绝） | 具名拒绝 |

拒绝是运行时的具名错误（CLI 退出码 1，含 `status: failed` 与原因），不是页面提示或文档约定；
`T09-verify/verify_candidates.py` 会在当前源码上重新触发这些拒绝路径。

## 性能口径（五模型一致）

- **CUBE**：含矩阵计算的算子，`ms = B·M·K·N ÷ 吞吐`；基准 B1/M1/K64/N64 = 1 ms ⇒ 默认 4096 MAC/ms。
- **VEC**：不含矩阵计算的算子，`ms = M·N ÷ 吞吐`；基准 M1/N64 = 1 ms ⇒ 默认 64 元素/ms。
- 连续按比例、**不向 64 取整**；不计带宽/并行/融合/通信/精度折扣，不是 TFLOPs，不另乘 2。
- 表内可编辑吞吐与 B/seqlen/kvcachelen/命中数等输入；真实 Excel 公式引用本组参数区与本表输入，改后重算。
- 每个算子行：分类、B/M/K/N（或 M/N）、工作量、成本状态、性能假设与维度映射（源码依据）。
- 层组末尾给单层小计与“单层小计×真实层数”；整网末尾按串行顺序求和各层组总计与入口/出口外围段，只乘一次。
- 维度守卫：符号与最终维度必须是 ≥0 的整数；负数/小数 → 未知，不产生负成本或静默截断。
- 吞吐守卫：数值且 >0 才计成本；≤0/空白/文本 → 未知。小数耗时列格式 `#,##0.000000`，不被显示成 0。
- 启用性能区时冻结窗格到 **C2**（A=黑盒标记、B=算子 id 常驻），横向滚到性能列仍能识别算子行。

### DeepSeek-V4-Flash-0731（`deepseek_v4_flash_0731_hf`）

- 当前候选：`c4a2f98acd52a8f40b08481a84771c58f276957c6bbd9acf2da3900d24e9c4a5`
- 目录：[`structure_reader/outputs/deepseek_v4_flash_0731_hf/c4a2f98acd52a8f40b08481a84771c58f276957c6bbd9acf2da3900d24e9c4a5/`](../outputs/deepseek_v4_flash_0731_hf/c4a2f98acd52a8f40b08481a84771c58f276957c6bbd9acf2da3900d24e9c4a5/)
- 覆盖：43 层 / 4 组：`swa-hash`(0,1)、`csa-hash`(2)、`hca-topk`(3,5…41)、`csa-topk`(4,6…42)；工作表 Prefill/Decode。
- 范围与限制：只做静态源码/配置分析；无真实权重、不实例化模型、不做数值等价；`--onnx` 未重建。
性能：四层组 + 整网外围逐算子 CUBE/VEC；`qk`/`av` 的 KV 轴是 `Lkv`（无压缩条目）；
`kv_b_proj` 对全部 `Lkv` 个缓存位置展开；mask 与 main/compress RoPE 在 `model.forward` 只建一次，
层组内为引用行、整网在外围各收一次。MoE 专家计算按总路由分配数 `B·seqlen·K`，命中专家权重切片用
本表 `N_hit`（默认未知，守卫 `≤min(E,B·seqlen·K)`、非空 `≥K`）。默认 Prefill 因 `N_hit` 未给定而
没有完整总计，只给已知部分小计与缺口数；填相容 `N_hit` 后才给数值。

| 产物 | SHA256 | 字节 |
| --- | --- | --- |
| [`viewer/structure.html`](../outputs/deepseek_v4_flash_0731_hf/c4a2f98acd52a8f40b08481a84771c58f276957c6bbd9acf2da3900d24e9c4a5/viewer/structure.html) | `f4be84b33c0dcc7cc7a18492007016b60e783ad73f34647ea34254ba6a7f9276` | 2694072 |
| [`tables/decoder_variants.xlsx`](../outputs/deepseek_v4_flash_0731_hf/c4a2f98acd52a8f40b08481a84771c58f276957c6bbd9acf2da3900d24e9c4a5/tables/decoder_variants.xlsx) | `2dabd3af336e08b59cc847b7cc8d5c89fc8bc46244abf3a685a0077cc365b05e` | 401381 |
| `data/layer_structures.json` | `f9c444115ecdd4333847df77d4aa440b5090f3ce5f64eb68fd6a30ce9e2f236c` | 793560 |
| `data/network_overview.json` | `2255b767cc124ad33500ac0e08ecedddf51fc486c3fc17ae4f2dfd29485f7b6a` | 53152 |
| `reports/source_manifest.json` | `4e2a03e02c4ef9bdd819734fdfe5b55165ca64df02382433eaedaefd749023c3` | 11568 |
| `run_manifest.json` | `8cfcd16aaa81aed224045d7bdc1304174647064235f2c8bfaac559ddde23f312` | 8376 |
| `README.md` | `43842a3676d20d06719c6c40e116b352225ee51360db56c6adb7246eecda2f0f` | 1383 |

### GLM-5.3（`glm_5_3_hf`）

- 当前候选：`1ff1a8282b8b8675352231ca4bc499db9e20ce371362d634a8d12427a0644029`
- 目录：[`structure_reader/outputs/glm_5_3_hf/1ff1a8282b8b8675352231ca4bc499db9e20ce371362d634a8d12427a0644029/`](../outputs/glm_5_3_hf/1ff1a8282b8b8675352231ca4bc499db9e20ce371362d634a8d12427a0644029/)
- 覆盖：78 层 / 3 组：`full-dense`(0-2)、`shared-moe`(57 层)、`full-moe`(18 层)；工作表 Prefill/Decode。
- 范围与限制：只做静态源码/配置分析；无真实权重、不实例化模型、不做数值等价；`--onnx` 未重建。
性能：三组层型（full-dense/shared-moe/full-moe）+ 整网外围逐算子；MLA `kv_b_proj` 对全部 `Lkv` 位置展开；
full 层 Indexer 每层算 q/k 打分 top-k，shared 层复用上一层 top-k（只把索引散进 mask，不重算 Indexer）；
MoE（E=256/K=8，sigmoid+组选择）专家计算按总分配数 `Ttok·K`；依赖命中数的“每命中专家扫描分配矩阵”
用本表 `N_hit`（默认未知）。GLM-5.3 结构没有独立的命中专家权重切片算子，权重读取不另收（用户规则不含带宽）。

| 产物 | SHA256 | 字节 |
| --- | --- | --- |
| [`viewer/structure.html`](../outputs/glm_5_3_hf/1ff1a8282b8b8675352231ca4bc499db9e20ce371362d634a8d12427a0644029/viewer/structure.html) | `03173ef554377cc6e2b6d8ff14854e8c1f4a1d0d9f4ca406cd9c449545aa1e6c` | 1589793 |
| [`tables/decoder_variants.xlsx`](../outputs/glm_5_3_hf/1ff1a8282b8b8675352231ca4bc499db9e20ce371362d634a8d12427a0644029/tables/decoder_variants.xlsx) | `2aeeed05be4954e8ff31b32885b1dd44c991e228e9dce56c8df07f108e464647` | 265429 |
| `data/layer_structures.json` | `7b6de5bf7e0a567bfb8607a976da42db0dabd4fdedd200a28c6d8331fb25940a` | 505488 |
| `data/network_overview.json` | `905b117daea86a61367fc18d7549220472f30ff6a71b5ccea79a94f3b78aa503` | 51809 |
| `data/effective_config.json` | `12b6782f9d8d7b46f1a9c0d2e3c6dfcae73bdbdc364a5b6425742927e03a574a` | 66358 |
| `reports/source_manifest.json` | `8b1f65ae2b5019306c1eea52b6fdb5caecb6457a8c51d244f341ed42678f675a` | 11526 |
| `run_manifest.json` | `8b9b2d1612afecc250638812123ad0a666d5514c8b8fc4944254c9fc2289dd56` | 9181 |
| `README.md` | `e7abf4ca99a6c5707da9e3ef121f85d456e016834aa6301e6b061db0deb787fb` | 1803 |

### GLM-5.3-Flash（`glm_5_3_flash_hf`）

- 当前候选：`e0a72bc340d038c20926ceae09ed648bfad69bf20eed512eecd475db1b0a0cc8`
- 目录：[`structure_reader/outputs/glm_5_3_flash_hf/e0a72bc340d038c20926ceae09ed648bfad69bf20eed512eecd475db1b0a0cc8/`](../outputs/glm_5_3_flash_hf/e0a72bc340d038c20926ceae09ed648bfad69bf20eed512eecd475db1b0a0cc8/)
- 覆盖：45 层 / 3 组：`kda-dense`(0-2)、`dsa-moe`(3,7…43，11 层)、`kda-moe`(31 层)；工作表 Prefill/Decode。
- 范围与限制：只做静态源码/配置分析；无真实权重、不实例化模型、不做数值等价；`--onnx` 未重建。
性能：KDA（depthwise causal conv1d：`B_eff=B·3·NhL·DhL, M=seqlen, K=Kconv, N=1`）与 gated delta-rule
逐 token 数学核（`mem=A^T·k_t`、`o=S_t^T·q_t` 两个矩阵-向量 + `S_t=A+k_t·δ_tᵀ` rank-1 更新，各一行，
总 MAC=3·B·NhL·seqlen·DhL·DhL；chunk 的 64 长度 pad/分块改写不另计，不声称 chunk 实际 MAC 与递推相同）；
DSA indexer 池化/打分/选中池展开/tail/pad `Kout` 分别表达，主 QK/AV 仍是完整 `Lkv` + 稀疏 mask；
4 路 mHC 的 fn/comb 混合/坍缩逐项计；MoE（E=288/K=8，SwiGLU clamp）按总分配数，`N_hit` 同前。

| 产物 | SHA256 | 字节 |
| --- | --- | --- |
| [`viewer/structure.html`](../outputs/glm_5_3_flash_hf/e0a72bc340d038c20926ceae09ed648bfad69bf20eed512eecd475db1b0a0cc8/viewer/structure.html) | `0ef8cb280e5287fd926bdae9f62efd03ef57039dbdb5497194597b2b86d42359` | 1872018 |
| [`tables/decoder_variants.xlsx`](../outputs/glm_5_3_flash_hf/e0a72bc340d038c20926ceae09ed648bfad69bf20eed512eecd475db1b0a0cc8/tables/decoder_variants.xlsx) | `fe648a311a70e8e06b22220700c3522862b77f4b0f47172dfec19bd909f28d78` | 294467 |
| `data/layer_structures.json` | `79610b97ab4e625618f64660d908d948a4e38e9743126b5deee52f2ae155535d` | 572480 |
| `data/network_overview.json` | `d2cb687ca58e4eae348480de480b1c8d1f73b20fddbfa5c89d8e6cf2a94640f7` | 44051 |
| `data/effective_config.json` | `f585adc13287374e0f67ed6147c83a7b8226d2a1bfe5d6dc89233c5b31be3c5e` | 114909 |
| `reports/source_manifest.json` | `e2ef6393800464b1e8b9006b9ced9f5f6fc1ec379b2e003f6a6d50a271143d4a` | 11278 |
| `run_manifest.json` | `0bad6ea4c73b18e33588d1843164fa0f46b6e445f8a633b07f7484adb9aa0a20` | 8642 |
| `README.md` | `de172568c2400375d46a755cf682a3ababc6a79b4e08cf11fdfce4e5079c3a87` | 2280 |

### Qwen3.8-Flash-Next（`qwen_3_8_flash_next_hf`）

- 当前候选：`7b5bee0e46a04d56bffe1a79dbf1d0d08bdb47271d35b0204b70feafa8e8c17b`
- 目录：[`structure_reader/outputs/qwen_3_8_flash_next_hf/7b5bee0e46a04d56bffe1a79dbf1d0d08bdb47271d35b0204b70feafa8e8c17b/`](../outputs/qwen_3_8_flash_next_hf/7b5bee0e46a04d56bffe1a79dbf1d0d08bdb47271d35b0204b70feafa8e8c17b/)
- 覆盖：48 层 / 3 组：`gdn-moe`(35 层)、`gdn-ple-moe`(layer 1)、`qsa-moe`(3,7…47，12 层)；工作表 Prefill/Decode。
- 范围与限制：只做静态源码/配置分析；无真实权重、不实例化模型、不做数值等价；`--onnx` 未重建。
性能：GDN（depthwise conv：`B_eff=B·Dqkv, M=seqlen, K=Kconv, N=1`；递推三矩阵项同上，状态 `Dk×DvH`）与
QSA 逐 batch/逐 query indexer：`V_q=P+j+1`、`N_q=V_q//Kc`、`B_q=min(Btop,N_q)`、`tail=V_q%Kc`，
整次查询用**单 batch 精确和** `ΣNq/ΣBq/Σtail`（整除前缀闭式）再乘 B；主 QK/AV 仍是完整 `Lkv` 矩阵 +
稀疏 mask，不使用 `Ksel_last·S` 或某个固定 j；逻辑选中数与 pad 后固定宽度 `Kout=Ki+Kc-1` 分开表达。
PLE（仅 layer 1）的逐 token 点积门、dilation=3 depthwise 短卷积、n-gram 查表逐项计；
MoE（E=512/K=10，softmax）按总分配数，`N_hit` 同前；网络末端没有 Final RMSNorm（`net_mixer_mix` 收拢后直接 LM Head）。

| 产物 | SHA256 | 字节 |
| --- | --- | --- |
| [`viewer/structure.html`](../outputs/qwen_3_8_flash_next_hf/7b5bee0e46a04d56bffe1a79dbf1d0d08bdb47271d35b0204b70feafa8e8c17b/viewer/structure.html) | `a0efb25e4f20372283274cbde30f8ecb00979c9fb54d4d863fefe24073b7b243` | 2101188 |
| [`tables/decoder_variants.xlsx`](../outputs/qwen_3_8_flash_next_hf/7b5bee0e46a04d56bffe1a79dbf1d0d08bdb47271d35b0204b70feafa8e8c17b/tables/decoder_variants.xlsx) | `500d2685904c2cf500ea15c54e86236cbe98462bca2afe34cc8dfd2bff404813` | 315873 |
| `data/layer_structures.json` | `f24d14c2b17a7024332b0c23cc60827ccced1884e7d7b7fdf341324fd822b30c` | 627289 |
| `data/network_overview.json` | `c7ddf7a21b193370c2a979a697006c3d5ee654339b344c0c511750d2248992a9` | 51037 |
| `data/effective_config.json` | `1a6f8500faf0574639dbb67f2c5eeda9a7794f73cc99081bed0040e83c740bfa` | 30654 |
| `reports/source_manifest.json` | `3744979a857ba1c28fac35679bd14375f24eb61a540a7886b8bb3be3ad5bb668` | 11588 |
| `run_manifest.json` | `c6297c5bce5f9d5688220cbae875eb34a6bed8166a83a86b800298c7643ac586` | 8795 |
| `README.md` | `5c3ffe8cf97b531a4e98f5669d39784875a6a84a2cff6b26c38f3672f446b03d` | 2459 |

### HiDream-O1-Image（`hidream_o1_image_hf`）

- 当前候选：`b4cc406ace15beb62637cad18aa2b6f530e98d129cd969b32400ea23831d34de`
- 目录：[`structure_reader/outputs/hidream_o1_image_hf/b4cc406ace15beb62637cad18aa2b6f530e98d129cd969b32400ea23831d34de/`](../outputs/hidream_o1_image_hf/b4cc406ace15beb62637cad18aa2b6f530e98d129cd969b32400ea23831d34de/)
- 覆盖：单次 forward：`uit-decoder` 36 层 + `vit-tower` 27 blocks（Kref>0 条件执行）；工作表 Generation。
- 范围与限制：只做静态源码/配置分析；无真实权重、不实例化模型、不做数值等价；`--onnx` 未重建。
性能：单次 forward 的 36 层 decoder（两趟注意力：AR 子集 + 全序列 non-causal 都计入）+ 27 blocks 视觉塔 +
整网一次性任务。视觉塔与参考图分支**先判 `Kref`**：Kref=0 时分支行按条件计 0（即使 `S_i` 空白），
层组总计与整网串行贡献也按条件直接记 0；Kref>0 时要求 `Svis≥1、S_i≥1、Kref·S_i=Svis`（逐图等长），
缺失/不等长 → 未知，不把不同长度平均成单值。`net_vmerge_norm` 在 view 之前（`Svis×Hv`），DeepStack 的
3 个 tap 读取与 decoder 前 3 层注入不在 36/27 层模板里重复计，整网 `net_deepstack_inject` 一次性收费。

| 产物 | SHA256 | 字节 |
| --- | --- | --- |
| [`viewer/structure.html`](../outputs/hidream_o1_image_hf/b4cc406ace15beb62637cad18aa2b6f530e98d129cd969b32400ea23831d34de/viewer/structure.html) | `1dad273c41d4aa12bbc4e12c01491f188289f1bb75bde45fec3600cbb1b98837` | 666679 |
| [`tables/decoder_variants.xlsx`](../outputs/hidream_o1_image_hf/b4cc406ace15beb62637cad18aa2b6f530e98d129cd969b32400ea23831d34de/tables/decoder_variants.xlsx) | `9a028e7a207d9dd73b863fbf64a22460a7e19b5880ce108634cb95d062bd159e` | 70729 |
| `data/layer_structures.json` | `06de5e9370f4966b1fc4ec349a67d8a4111fd2d0e4203ee67f8ca503d657b8e2` | 151456 |
| `data/network_overview.json` | `2a553b930b24817699ba87989e7d065ba1a87a9b01fec6014c65c1bf2fb75190` | 74354 |
| `data/effective_config.json` | `6b06eb8e25c64f725e19174f6eeda371825c07e45136f307ceae5efc54d38700` | 13185 |
| `reports/source_manifest.json` | `3cc2519ae5c5b59aff79648aa231d46ef3d5b02c4da0974e823d46e0b9bc5346` | 10806 |
| `run_manifest.json` | `8dee3e4be509c3a4ab39ee943a0c20f968daacd45afb748d98f0443fcae608d9` | 8801 |
| `README.md` | `4ebd2edebb29cbb743c6087d3c24eb81370967d1de93aff0df1d451eac1181b0` | 1882 |

## 历史正式候选（保留，不作为当前）

工具源码在 T10/T11 发生了变化，因此下面这些更早的候选身份不能再当作当前实现证据；目录与内容原样保留：

| 模型 | 历史 `analysis_id` | 说明 |
| --- | --- | --- |
| DeepSeek-V4-Flash-0731 | `dcbf7eae3a26a0f7745c76b884bf55ae8406f7fe6d5631151075516ef09f8280` | T10 已验收的 CUBE/VEC 首版（607 测试/10 组两页重算）；T11 源码变化后保留 |
| DeepSeek-V4-Flash-0731 | `c3fd4296377dcf42ef682ebbebd00f013595a17cde0ca0c3e6f9a9c472e22ff9` | T05–T08 阶段验收候选 |
| GLM-5.3 | `abefb101988c519dc81f5cf60aa27553e312cd0038d799701e4c460415456621` | T05–T08 阶段验收候选 |
| GLM-5.3-Flash | `29f82a3e5481c971cdf36bb61f17ac60ef20036d86bdce33b49343bb09f41003` | T05–T08 阶段验收候选 |
| Qwen3.8-Flash-Next | `c921712221160320176fe73003fc02b78c6c53a0d9c8bf895908b0250bef8454` | T05–T08 阶段验收候选 |
| HiDream-O1-Image | `5423a22f4a8c781c147c3bc80d4005a25417869f29372910d11aca7224663b91` | T05–T08 阶段验收候选 |

T09/T10/T11 期间的中间/返修候选同样保留在各自 `outputs/<model_key>/` 下，回退或对照时以各自
`run_manifest.json`/`reports/source_manifest.json` 的 SHA 为准，不按目录时间猜“最新”。

## 未完成与未验证范围

- 本票未重建 DeepSeek 的模块级 ONNX：本机缺 `onnx/onnxsim/onnxruntime`。能力与代码入口保留，
  历史含 ONNX 的候选保留；不得把旧 ONNX 目录算作当前候选产物。
- 全部结论都来自静态源码/配置分析：不下载权重、不实例化/运行模型、不做数值等价。
- 性能是用户确认的**归一化粗估基准**（4096 MAC/ms、64 元素/ms），不是设备实测；量级很大是基准本身的结果。
- MoE 的命中专家数 `N_hit` 是“各层统一命中数”的显式可编辑假设，不推断实际路由分布；
  默认未给定时相关层组小计与整网总计为未知（给已知部分小计与缺口数）。
- 原生重算与渲染证据覆盖五个工作簿的全部 9 个工作表（见 `T11-cube-vec-verify/`）；
  没有 Microsoft Excel GUI 验收，也没有用户阅读体验验收。

T09 阶段记录的五模型“成本待提供”状态已被 T10/T11 的用户 CUBE/VEC 规则取代：当前五份工作簿的成本列
都是可重算的用户规则耗时（未知项显示“未知（原因）”），不再有整列“待提供”的成本。

## 机器可读索引

下面 JSON 是 `T09-verify/verify_candidates.py` 的校验对象；修改候选时必须同步更新（校验会失败而不是静默通过）。

<!-- T09-DELIVERY-INDEX:BEGIN -->
```json
{
 "schema": "t09_delivery_index_v1",
 "generated_at": "2026-10-29",
 "head": "947ae78f120279403ec0a4c539e99d3d8869e1d6",
 "working_tree_uncommitted": true,
 "build_command": "build-hf --model <model_key> --all-variants --archify-layers --json",
 "models": [
  {
   "model_key": "deepseek_v4_flash_0731_hf",
   "analysis_id": "c4a2f98acd52a8f40b08481a84771c58f276957c6bbd9acf2da3900d24e9c4a5",
   "output_dir": "structure_reader/outputs/deepseek_v4_flash_0731_hf/c4a2f98acd52a8f40b08481a84771c58f276957c6bbd9acf2da3900d24e9c4a5",
   "display_name": "DeepSeek-V4-Flash-0731 (HF transformers)",
   "model_revision": null,
   "source_revision": "536ecc007387a50e77603bb5d92100e9b07514cc",
   "config_path": "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json",
   "groups": {
    "csa-hash": [
     2
    ],
    "csa-topk": [
     4,
     6,
     8,
     10,
     12,
     14,
     16,
     18,
     20,
     22,
     24,
     26,
     28,
     30,
     32,
     34,
     36,
     38,
     40,
     42
    ],
    "hca-topk": [
     3,
     5,
     7,
     9,
     11,
     13,
     15,
     17,
     19,
     21,
     23,
     25,
     27,
     29,
     31,
     33,
     35,
     37,
     39,
     41
    ],
    "swa-hash": [
     0,
     1
    ]
   },
   "layer_count": 43,
   "sheets": [
    "Prefill",
    "Decode"
   ],
   "artifacts": {
    "README.md": {
     "sha256": "43842a3676d20d06719c6c40e116b352225ee51360db56c6adb7246eecda2f0f",
     "bytes": 1383
    },
    "data/layer_structures.json": {
     "sha256": "f9c444115ecdd4333847df77d4aa440b5090f3ce5f64eb68fd6a30ce9e2f236c",
     "bytes": 793560
    },
    "data/network_overview.json": {
     "sha256": "2255b767cc124ad33500ac0e08ecedddf51fc486c3fc17ae4f2dfd29485f7b6a",
     "bytes": 53152
    },
    "reports/source_manifest.json": {
     "sha256": "4e2a03e02c4ef9bdd819734fdfe5b55165ca64df02382433eaedaefd749023c3",
     "bytes": 11568
    },
    "run_manifest.json": {
     "sha256": "8cfcd16aaa81aed224045d7bdc1304174647064235f2c8bfaac559ddde23f312",
     "bytes": 8376
    },
    "tables/decoder_variants.xlsx": {
     "sha256": "2dabd3af336e08b59cc847b7cc8d5c89fc8bc46244abf3a685a0077cc365b05e",
     "bytes": 401381
    },
    "viewer/structure.html": {
     "sha256": "f4be84b33c0dcc7cc7a18492007016b60e783ad73f34647ea34254ba6a7f9276",
     "bytes": 2694072
    }
   }
  },
  {
   "model_key": "glm_5_3_hf",
   "analysis_id": "1ff1a8282b8b8675352231ca4bc499db9e20ce371362d634a8d12427a0644029",
   "output_dir": "structure_reader/outputs/glm_5_3_hf/1ff1a8282b8b8675352231ca4bc499db9e20ce371362d634a8d12427a0644029",
   "display_name": "GLM-5.3 (zai-org/GLM-5.3, HF transformers glm_moe_dsa)",
   "model_revision": "ba32f8c19c8738e2c57c71798c601d477a7bf844",
   "source_revision": "536ecc007387a50e77603bb5d92100e9b07514cc",
   "config_path": "models/zai-org/GLM-5.3/config.json",
   "groups": {
    "full-dense": [
     0,
     1,
     2
    ],
    "full-moe": [
     6,
     10,
     14,
     18,
     22,
     26,
     30,
     34,
     38,
     42,
     46,
     50,
     54,
     58,
     62,
     66,
     70,
     74
    ],
    "shared-moe": [
     3,
     4,
     5,
     7,
     8,
     9,
     11,
     12,
     13,
     15,
     16,
     17,
     19,
     20,
     21,
     23,
     24,
     25,
     27,
     28,
     29,
     31,
     32,
     33,
     35,
     36,
     37,
     39,
     40,
     41,
     43,
     44,
     45,
     47,
     48,
     49,
     51,
     52,
     53,
     55,
     56,
     57,
     59,
     60,
     61,
     63,
     64,
     65,
     67,
     68,
     69,
     71,
     72,
     73,
     75,
     76,
     77
    ]
   },
   "layer_count": 78,
   "sheets": [
    "Prefill",
    "Decode"
   ],
   "artifacts": {
    "README.md": {
     "sha256": "e7abf4ca99a6c5707da9e3ef121f85d456e016834aa6301e6b061db0deb787fb",
     "bytes": 1803
    },
    "data/effective_config.json": {
     "sha256": "12b6782f9d8d7b46f1a9c0d2e3c6dfcae73bdbdc364a5b6425742927e03a574a",
     "bytes": 66358
    },
    "data/layer_structures.json": {
     "sha256": "7b6de5bf7e0a567bfb8607a976da42db0dabd4fdedd200a28c6d8331fb25940a",
     "bytes": 505488
    },
    "data/network_overview.json": {
     "sha256": "905b117daea86a61367fc18d7549220472f30ff6a71b5ccea79a94f3b78aa503",
     "bytes": 51809
    },
    "reports/source_manifest.json": {
     "sha256": "8b1f65ae2b5019306c1eea52b6fdb5caecb6457a8c51d244f341ed42678f675a",
     "bytes": 11526
    },
    "run_manifest.json": {
     "sha256": "8b9b2d1612afecc250638812123ad0a666d5514c8b8fc4944254c9fc2289dd56",
     "bytes": 9181
    },
    "tables/decoder_variants.xlsx": {
     "sha256": "2aeeed05be4954e8ff31b32885b1dd44c991e228e9dce56c8df07f108e464647",
     "bytes": 265429
    },
    "viewer/structure.html": {
     "sha256": "03173ef554377cc6e2b6d8ff14854e8c1f4a1d0d9f4ca406cd9c449545aa1e6c",
     "bytes": 1589793
    }
   }
  },
  {
   "model_key": "glm_5_3_flash_hf",
   "analysis_id": "e0a72bc340d038c20926ceae09ed648bfad69bf20eed512eecd475db1b0a0cc8",
   "output_dir": "structure_reader/outputs/glm_5_3_flash_hf/e0a72bc340d038c20926ceae09ed648bfad69bf20eed512eecd475db1b0a0cc8",
   "display_name": "GLM-5.3-Flash (zai-org/GLM-5.3-Flash, HF transformers glm5_next 文本主干)",
   "model_revision": "46a050bec13002ad29ee795c16e61cf4fa3c4a9c",
   "source_revision": "536ecc007387a50e77603bb5d92100e9b07514cc",
   "config_path": "models/zai-org/GLM-5.3-Flash/config.json",
   "groups": {
    "dsa-moe": [
     3,
     7,
     11,
     15,
     19,
     23,
     27,
     31,
     35,
     39,
     43
    ],
    "kda-dense": [
     0,
     1,
     2
    ],
    "kda-moe": [
     4,
     5,
     6,
     8,
     9,
     10,
     12,
     13,
     14,
     16,
     17,
     18,
     20,
     21,
     22,
     24,
     25,
     26,
     28,
     29,
     30,
     32,
     33,
     34,
     36,
     37,
     38,
     40,
     41,
     42,
     44
    ]
   },
   "layer_count": 45,
   "sheets": [
    "Prefill",
    "Decode"
   ],
   "artifacts": {
    "README.md": {
     "sha256": "de172568c2400375d46a755cf682a3ababc6a79b4e08cf11fdfce4e5079c3a87",
     "bytes": 2280
    },
    "data/effective_config.json": {
     "sha256": "f585adc13287374e0f67ed6147c83a7b8226d2a1bfe5d6dc89233c5b31be3c5e",
     "bytes": 114909
    },
    "data/layer_structures.json": {
     "sha256": "79610b97ab4e625618f64660d908d948a4e38e9743126b5deee52f2ae155535d",
     "bytes": 572480
    },
    "data/network_overview.json": {
     "sha256": "d2cb687ca58e4eae348480de480b1c8d1f73b20fddbfa5c89d8e6cf2a94640f7",
     "bytes": 44051
    },
    "reports/source_manifest.json": {
     "sha256": "e2ef6393800464b1e8b9006b9ced9f5f6fc1ec379b2e003f6a6d50a271143d4a",
     "bytes": 11278
    },
    "run_manifest.json": {
     "sha256": "0bad6ea4c73b18e33588d1843164fa0f46b6e445f8a633b07f7484adb9aa0a20",
     "bytes": 8642
    },
    "tables/decoder_variants.xlsx": {
     "sha256": "fe648a311a70e8e06b22220700c3522862b77f4b0f47172dfec19bd909f28d78",
     "bytes": 294467
    },
    "viewer/structure.html": {
     "sha256": "0ef8cb280e5287fd926bdae9f62efd03ef57039dbdb5497194597b2b86d42359",
     "bytes": 1872018
    }
   }
  },
  {
   "model_key": "qwen_3_8_flash_next_hf",
   "analysis_id": "7b5bee0e46a04d56bffe1a79dbf1d0d08bdb47271d35b0204b70feafa8e8c17b",
   "output_dir": "structure_reader/outputs/qwen_3_8_flash_next_hf/7b5bee0e46a04d56bffe1a79dbf1d0d08bdb47271d35b0204b70feafa8e8c17b",
   "display_name": "Qwen3.8-Flash-Next (Qwen/Qwen3.8-Flash-Next, HF transformers qwen4_exp 文本主干)",
   "model_revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
   "source_revision": "536ecc007387a50e77603bb5d92100e9b07514cc",
   "config_path": "models/Qwen/Qwen3.8-Flash-Next/config.json",
   "groups": {
    "gdn-moe": [
     0,
     2,
     4,
     5,
     6,
     8,
     9,
     10,
     12,
     13,
     14,
     16,
     17,
     18,
     20,
     21,
     22,
     24,
     25,
     26,
     28,
     29,
     30,
     32,
     33,
     34,
     36,
     37,
     38,
     40,
     41,
     42,
     44,
     45,
     46
    ],
    "gdn-ple-moe": [
     1
    ],
    "qsa-moe": [
     3,
     7,
     11,
     15,
     19,
     23,
     27,
     31,
     35,
     39,
     43,
     47
    ]
   },
   "layer_count": 48,
   "sheets": [
    "Prefill",
    "Decode"
   ],
   "artifacts": {
    "README.md": {
     "sha256": "5c3ffe8cf97b531a4e98f5669d39784875a6a84a2cff6b26c38f3672f446b03d",
     "bytes": 2459
    },
    "data/effective_config.json": {
     "sha256": "1a6f8500faf0574639dbb67f2c5eeda9a7794f73cc99081bed0040e83c740bfa",
     "bytes": 30654
    },
    "data/layer_structures.json": {
     "sha256": "f24d14c2b17a7024332b0c23cc60827ccced1884e7d7b7fdf341324fd822b30c",
     "bytes": 627289
    },
    "data/network_overview.json": {
     "sha256": "c7ddf7a21b193370c2a979a697006c3d5ee654339b344c0c511750d2248992a9",
     "bytes": 51037
    },
    "reports/source_manifest.json": {
     "sha256": "3744979a857ba1c28fac35679bd14375f24eb61a540a7886b8bb3be3ad5bb668",
     "bytes": 11588
    },
    "run_manifest.json": {
     "sha256": "c6297c5bce5f9d5688220cbae875eb34a6bed8166a83a86b800298c7643ac586",
     "bytes": 8795
    },
    "tables/decoder_variants.xlsx": {
     "sha256": "500d2685904c2cf500ea15c54e86236cbe98462bca2afe34cc8dfd2bff404813",
     "bytes": 315873
    },
    "viewer/structure.html": {
     "sha256": "a0efb25e4f20372283274cbde30f8ecb00979c9fb54d4d863fefe24073b7b243",
     "bytes": 2101188
    }
   }
  },
  {
   "model_key": "hidream_o1_image_hf",
   "analysis_id": "b4cc406ace15beb62637cad18aa2b6f530e98d129cd969b32400ea23831d34de",
   "output_dir": "structure_reader/outputs/hidream_o1_image_hf/b4cc406ace15beb62637cad18aa2b6f530e98d129cd969b32400ea23831d34de",
   "display_name": "HiDream-O1-Image（HiDream-ai/HiDream-O1-Image, Pixel-level Unified Transformer 单次生成网络）",
   "model_revision": "0b0901d99f200389e138c61946af1185f5f49a13",
   "source_revision": "2c2d29ff729e48f33e41f49edfdbd81d5ac103b4",
   "config_path": "models/HiDream/HiDream-O1-Image/config.json",
   "groups": {
    "uit-decoder": [
     0,
     1,
     2,
     3,
     4,
     5,
     6,
     7,
     8,
     9,
     10,
     11,
     12,
     13,
     14,
     15,
     16,
     17,
     18,
     19,
     20,
     21,
     22,
     23,
     24,
     25,
     26,
     27,
     28,
     29,
     30,
     31,
     32,
     33,
     34,
     35
    ],
    "vit-tower": [
     0,
     1,
     2,
     3,
     4,
     5,
     6,
     7,
     8,
     9,
     10,
     11,
     12,
     13,
     14,
     15,
     16,
     17,
     18,
     19,
     20,
     21,
     22,
     23,
     24,
     25,
     26
    ]
   },
   "layer_count": 36,
   "sheets": [
    "Generation"
   ],
   "artifacts": {
    "README.md": {
     "sha256": "4ebd2edebb29cbb743c6087d3c24eb81370967d1de93aff0df1d451eac1181b0",
     "bytes": 1882
    },
    "data/effective_config.json": {
     "sha256": "6b06eb8e25c64f725e19174f6eeda371825c07e45136f307ceae5efc54d38700",
     "bytes": 13185
    },
    "data/layer_structures.json": {
     "sha256": "06de5e9370f4966b1fc4ec349a67d8a4111fd2d0e4203ee67f8ca503d657b8e2",
     "bytes": 151456
    },
    "data/network_overview.json": {
     "sha256": "2a553b930b24817699ba87989e7d065ba1a87a9b01fec6014c65c1bf2fb75190",
     "bytes": 74354
    },
    "reports/source_manifest.json": {
     "sha256": "3cc2519ae5c5b59aff79648aa231d46ef3d5b02c4da0974e823d46e0b9bc5346",
     "bytes": 10806
    },
    "run_manifest.json": {
     "sha256": "8dee3e4be509c3a4ab39ee943a0c20f968daacd45afb748d98f0443fcae608d9",
     "bytes": 8801
    },
    "tables/decoder_variants.xlsx": {
     "sha256": "9a028e7a207d9dd73b863fbf64a22460a7e19b5880ce108634cb95d062bd159e",
     "bytes": 70729
    },
    "viewer/structure.html": {
     "sha256": "1dad273c41d4aa12bbc4e12c01491f188289f1bb75bde45fec3600cbb1b98837",
     "bytes": 666679
    }
   }
  }
 ],
 "historical_candidates": [
  {
   "model_key": "deepseek_v4_flash_0731_hf",
   "analysis_id": "dcbf7eae3a26a0f7745c76b884bf55ae8406f7fe6d5631151075516ef09f8280",
   "note": "T10 CUBE/VEC 首版（已验收）"
  },
  {
   "model_key": "deepseek_v4_flash_0731_hf",
   "analysis_id": "c3fd4296377dcf42ef682ebbebd00f013595a17cde0ca0c3e6f9a9c472e22ff9"
  },
  {
   "model_key": "glm_5_3_hf",
   "analysis_id": "abefb101988c519dc81f5cf60aa27553e312cd0038d799701e4c460415456621"
  },
  {
   "model_key": "glm_5_3_flash_hf",
   "analysis_id": "29f82a3e5481c971cdf36bb61f17ac60ef20036d86bdce33b49343bb09f41003"
  },
  {
   "model_key": "qwen_3_8_flash_next_hf",
   "analysis_id": "c921712221160320176fe73003fc02b78c6c53a0d9c8bf895908b0250bef8454"
  },
  {
   "model_key": "hidream_o1_image_hf",
   "analysis_id": "5423a22f4a8c781c147c3bc80d4005a25417869f29372910d11aca7224663b91"
  }
 ],
 "performance": {
  "engine": "cube_vec",
  "cube_default": 4096,
  "vec_default": 64,
  "verified": "T11-cube-vec-verify",
  "detail": "docs/tasks/model-structure-2026-10-28/T11-cube-vec-result.md"
 }
}
```
<!-- T09-DELIVERY-INDEX:END -->
