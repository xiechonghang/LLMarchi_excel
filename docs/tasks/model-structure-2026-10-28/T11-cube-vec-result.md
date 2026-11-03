# T11 结果：CUBE/VEC 性能表覆盖其余四模型并重建五模型交付

状态：**实现与验证完成，停止等待 Codex 复核**。未关闭 GitHub #12，未 commit/push，未改协调文件与自动化
（保持 PAUSED/45 分钟），未恢复旧交接会话。

## 接手核对

- 工作区 `/Users/xiechonghang/Workspace/LLMarchi_excel`；`main` HEAD `947ae78f120279403ec0a4c539e99d3d8869e1d6`
  （本票未产生提交，HEAD 未变）。
- 快照 `/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T11-cube-vec-baseline-20261029-9wmc952w/manifest.json`：
  706 文件；逐文件 SHA 比对**缺失 0**，变化 20、新增 54（[baseline-diff.json](T11-cube-vec-verify/baseline-diff.json)）。
  变化的 20 个里 4 个是快照后 Codex 写的协调文件（`README.md`/`execution.json`/`tickets.json`/
  `monitor-state.json`，本票只读），新增里 `T11-cube-vec-review/` 是 Codex 复核目录（本票未写）；根 `CONTEXT.md` 与
  `2026_10_17_workspace_structure_review.md` 的既有删除保持；旧候选、旧证据、旧单算子模板零覆盖。
- T10 已验收的 DeepSeek 候选 `dcbf7eae…`、T09 候选 `61b74593…` 与 T05–T08 历史候选全部保留（[candidate-closure.json](T11-cube-vec-verify/candidate-closure.json)）。

## 交付物（五模型当前候选）

| 模型 | `analysis_id` | 工作簿 | SHA256 | 字节 |
| --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | `c4a2f98a…` | [`tables/decoder_variants.xlsx`](../../../structure_reader/outputs/deepseek_v4_flash_0731_hf/c4a2f98acd52a8f40b08481a84771c58f276957c6bbd9acf2da3900d24e9c4a5/tables/decoder_variants.xlsx) | `2dabd3af…f05e` | 401,381 |
| GLM-5.3 | `1ff1a828…` | [`tables/decoder_variants.xlsx`](../../../structure_reader/outputs/glm_5_3_hf/1ff1a8282b8b8675352231ca4bc499db9e20ce371362d634a8d12427a0644029/tables/decoder_variants.xlsx) | `2aeeed05…4647` | 265,429 |
| GLM-5.3-Flash | `e0a72bc3…` | [`tables/decoder_variants.xlsx`](../../../structure_reader/outputs/glm_5_3_flash_hf/e0a72bc340d038c20926ceae09ed648bfad69bf20eed512eecd475db1b0a0cc8/tables/decoder_variants.xlsx) | `fe648a31…8d78` | 294,467 |
| Qwen3.8-Flash-Next | `7b5bee0e…` | [`tables/decoder_variants.xlsx`](../../../structure_reader/outputs/qwen_3_8_flash_next_hf/7b5bee0e46a04d56bffe1a79dbf1d0d08bdb47271d35b0204b70feafa8e8c17b/tables/decoder_variants.xlsx) | `500d2685…4813` | 315,873 |
| HiDream-O1-Image | `b4cc406a…` | [`tables/decoder_variants.xlsx`](../../../structure_reader/outputs/hidream_o1_image_hf/b4cc406ace15beb62637cad18aa2b6f530e98d129cd969b32400ea23831d34de/tables/decoder_variants.xlsx) | `9a028e7a…159e` | 70,729 |

完整 ID/全部产物 SHA 与统一入口见 [`structure_reader/docs/model_delivery_index.md`](../../../structure_reader/docs/model_delivery_index.md)
（已同步机器可读索引；`T09-verify/verify_candidates.py` 复核 `passed`，0 issue）。分层浏览入口仍为各自的
`viewer/structure.html`，结构页语义未变（DeepSeek HTML 仅 analysis_id 内嵌变化）。

## 规则覆盖与分类（每个实际算子都有规则）

完整逐算子清单见 [rules-coverage.json](T11-cube-vec-verify/rules-coverage.json)（分类、维度表达式、项数、
重复/条件、计费范围、来源说明、符号是否在本组声明）。

| 模型 | 实际算子/外围行 | CUBE | VEC | 容器 | 组/层数 |
| --- | --- | --- | --- | --- | --- |
| DeepSeek | 277 + 14 | 76 | 214 | 1 | swa-hash 2 / csa-hash 1 / hca-topk 20 / csa-topk 20，共 43 层 |
| GLM-5.3 | 193 + 8 | 47 | 153 | 1 | full-dense 3 / shared-moe 57 / full-moe 18，共 78 层 |
| GLM-5.3-Flash | 237 + 9 | 63 | 182 | 1 | kda-dense 3 / dsa-moe 11 / kda-moe 31，共 45 层 |
| Qwen3.8-Flash-Next | 268 + 9 | 67 | 209 | 1 | gdn-moe 35 / gdn-ple-moe 1 / qsa-moe 12，共 48 层 |
| HiDream-O1-Image | 原始 49 + 30（含 2 个循环容器）；Excel 78 个算子行 | 27 | 50 | 2 | uit-decoder 36 / vit-tower 27；Excel 去掉 2 个循环容器、加 1 个声明式 DeepStack 聚合行 = 78 行 |

缺少规则的算子 **0**，规则引用未声明符号 **0**；30 个手算锚点全部通过（CUBE 基准 B1/M1/K64/N64=1 ms、
B2/M3=6 ms、VEC M3/N100=4.6875 ms 不取整，以及 q_a_proj/kv_b_proj/qk/conv1d/scan/mhc/PLE/QSA/HiDream 真实算子）。

### 语义重点与源码依据

- **GLM-5.3**：MLA `kv_b_proj` 对全部 `Lkv` 个缓存位置展开；`qk`/`av` 的 KV 轴是 `Lkv`（无压缩条目/滑窗）。
  full 层 Indexer 每层算 q/k 打分 top-k；shared 层复用上一层 top-k（`g_index_mask_scatter`），不重算 Indexer。
  `g_ix_sum` 按“每个 (batch,query) 一份 `[Ih,Lidx]`、M=1”的逐 query 收缩记几何（B_eff=B·seqlen）。
  MoE（E=256/K=8）专家计算按总分配数 `Ttok·K`；只有“每命中专家扫描分配矩阵”一项用 `N_hit`。
- **GLM-5.3-Flash**：KDA depthwise causal conv1d 几何 `B_eff=B·3·NhL·DhL, M=seqlen, K=Kconv, N=1`；
  delta-rule 递推按逐 token 数学核拆三行（`mem=A^T·k_t`、`o=S_t^T·q_t` 两个矩阵-向量
  `B_eff=B·NhL·seqlen, M=1, K=N=DhL`；`S_t=A+k_t·δ_tᵀ` rank-1 `M=DhL,K=1,N=DhL`），总 MAC=3·B·NhL·seqlen·DhL²；
  chunk 的 64 长度 pad/分块改写不另计，也不声称 chunk 实际 MAC 与递推相同。DSA indexer 的池化/打分/选中池/
  tail/pad `Kout` 分别表达，主 QK/AV 仍是完整 `Lkv` + 稀疏 mask；`g_dsa_ix_sum` 同样是逐 query `M=1` 收缩。
- **Qwen3.8-Flash-Next**：GDN conv 几何 `B_eff=B·Dqkv, M=seqlen, K=Kconv, N=1`；递推三矩阵项同上（状态 `Dk×DvH`）。
  QSA indexer 在源码里**逐 batch、逐 query 循环**（`modeling_qwen4_exp.py:721`）：`V_q=P+j+1`、`N_q=V_q//Kc`、
  `B_q=min(Btop,N_q)`、`tail=V_q%Kc`；整次 S 个查询用**单 batch 精确和** `ΣNq/ΣBq/Σtail`（整除前缀闭式）再乘 B
  （13 组边界枚举核对，含 2047/2048/4092/4096 历史与 512 块上界前后）。主 QK/AV 仍是完整 `Lkv` 矩阵 +
  稀疏 mask，不用 `Ksel_last·S` 冒充整网；逻辑选中数与 pad 后固定宽度 `Kout=Ki+Kc-1` 分开表达。
  PLE 的逐 token 点积门 `B_eff=B·seqlen·Hc, M=1, K=H, N=1`；dilation=3 depthwise 短卷积按真实核长/感受野。
- **HiDream-O1-Image**：单次 forward 的 36 层 decoder 两趟注意力（AR 子集 + 全序列 non-causal）都计入；
  视觉塔与参考图分支**先判 `Kref`**：Kref=0 时分支行按条件计 0（即使 `S_i` 空白），层组总计与串行贡献也按条件
  直接记 0；Kref>0 要求 `Svis≥1、S_i≥1、Kref·S_i=Svis`，缺失/不等长 → 未知。`net_vmerge_norm` 在 view **之前**
  （`Svis×Hv`，不是 `Svism×Hv`）。DeepStack 的 3 个 tap 读取与 decoder 前 3 层注入不在 36/27 层模板里重复计，
  由声明式整网行 `net_deepstack_inject` 一次性收费（2 个逻辑工作张量各一行）。
- **MoE 统一假设**：`N_hit` 是“各层统一命中数”的显式可编辑输入（默认未知）；守卫 `N_hit ≤ min(E,B·seqlen·K)`、
  非空调用 `N_hit ≥ K`（TopK 为 K）或 `≥1`（Hash，DeepSeek）；不满足 → 未知，不给不可实现的完整总计。
  四个新模型的结构没有独立的“命中专家权重切片”算子，专家权重读取不另收（CUBE 只计 MAC；用户规则不含带宽）。

## 输入填写方式

每页顶部“性能配置与场景输入”（黄色 D 列可改）后紧接参数区/算子区；CUBE/VEC 吞吐、`B`/`seqlen`/`kvcachelen`、
`N_hit` 都在这一区；HiDream 另有 `image_height/image_width/prompt_tokens/ref_images/ref_patch_tokens/
cond_vision_tokens/cond_image_tokens/batch_size`。改任一项由 Excel 重算：维度 → 工作量 → 耗时 → 层组小计 →
串行汇总；`N_hit` 的参数字行引用同一输入单元格（不再有“参数区未知、输入区有值”两套口径）。

## 实际验证

1. **规则/几何**（[rules-coverage.json](T11-cube-vec-verify/rules-coverage.json)）：覆盖 0 缺口/0 未声明符号；
   30 个手算锚点（独立整数算术，不读 Excel）；QSA 13 组逐 query 枚举与产品闭式、以及
   `/private/tmp/llmarchi-qsa-sums-reference.json` 的独立结果全部一致（[qsa_sums] 段）。
2. **原生 LibreOffice 重算**（[verify-workbook.json](T11-cube-vec-verify/verify-workbook.json)；
   改表脚本 [verify_native_edits_t11.mjs](T11-cube-vec-verify/verify_native_edits_t11.mjs)）：
   - 五本工作簿全部 **9 个工作表**（DeepSeek/GLM53/Flash/Qwen 各 Prefill+Decode，HiDream Generation）逐页检查：
     27 列、每行分类、冻结 C2、成本列标题写明“默认 4096/64、实际按表内吞吐”、单层小计 `COUNT+SUM` 守卫、
     串行 F/G/H 语义、无 Excel 错误字面量；每页渲染新性能区/层组小计/串行汇总（36 张，
     [renders/](T11-cube-vec-verify/renders/)，含算子 id 对照图）。
   - 每模型 116 项共享数值/守卫检查全部通过：默认手算锚点（GLM53 `q_a_proj` 24,576、`kv_b_proj` 28,672、
     `qk` 256、`ix_sum` 0.5、`moe_router` 3,072；Flash KDA scan 三行各 2,048、conv1d 192、mHC mix 128、
     dense gate 98,304；Qwen qkv_proj 51,200、conv 80、scan 项 1,536、QSA `ix_pool` 48/`ix_scores` 0.75、
     `qk` 96、PLE gate 20；HiDream `q_proj` 4,718,592、`qk_full` 1,327,104、`gate_proj` 14,155,776、
     `net_patch_lin1` 786,432、`net_final_lin` 3,538,944；DeepSeek T10 锚点 8,192/1,034,240/512）。
   - 改表：seqlen=4096（GLM53 `q_a_proj` 12,582,912；DeepSeek 4,194,304）、吞吐 2048 成本翻倍、吞吐 0/文本未知、
     清空 seqlen 未知、H=-1/1.5 未知且无负成本；Decode 页改 seqlen=32 也重算（GLM53 98,304、
     Flash `kv_b_proj` 16,908,288、Qwen 主 QK 用完整 `Lkv=4128`）。
   - QSA：P0/S1、P3/S2（跨 4 token 块）、P2047/S4、P2048/S4、P4092/S4、P4096/S1、P0/S13 逐组核对
     `ix_pool/ix_tail/ix_scores`（单 batch 枚举和）与主 QK（完整 `Lkv`）；`B=2` 时逐 query 项翻倍。
   - MoE：GLM53/Flash/Qwen `N_hit=K`（8/10）相容 → 命中扫描数值、整网总计变数值；`N_hit=999` 不相容 → 未知。
   - HiDream：Kref=0（含 `S_i`/`Svis` 空白）分支 0 且整网总计数值；Kref=1 等长 → `v_norm1` 18,432、
     `net_vmerge_norm` 18,432、`net_vmerge_fc1` 1,327,104、DeepStack 55,296+49,152；Kref=1 但 `S_i` 空白、
     Kref=2 不等长、Kref 空白、Kref=0 但 `Svis>0` 均 → 整网总计未知（条件未知不当 False）。
3. **T10 数据/成本公式回归**（[regression.json](T11-cube-vec-verify/regression.json)）：
   - DeepSeek 新旧默认原生重算副本逐算子比较 **279 个成本数值全部相同**（T10 已验收口径未变）。
   - 工作簿逐单元格差异仅 18 个：10 个成本列标题（每页 5 个算子区）+ 8 个条件算子 `hca_block_bias`
     的维度/工作量/状态公式（Prefill/Decode 各 V/X/Y/Z 四格）。后者来自共享导出器“先判条件再算维度”的
     改造：有效场景成本不变，false 分支维度显示 0；**这不是“只有表头/冻结窗格变化”**，报告与索引已如实区分。
   - 其余四模型 `layer_structures.json`/`network_overview.json` 与上一候选逐字节相同；工作簿字节按预期变化。
4. **候选闭包**：五个新候选 `analysis_id` 由 `identity` 复算 = 目录名；deliverables SHA 一致；
   工具清单与冻结代码零漂移；T10 `dcbf7eae…`（工作簿 `d526126e…` 不变）与 T09 `61b74593…`
   （工作簿 `2da0caf1…` 不变）保留。
5. **测试**（[tests-final.json](T11-cube-vec-verify/tests-final.json)）：全量 `discover` **2027 项**，失败 6
   （均为 `test_hf_decoder_diagram.py` 旧布局断言，T09 已登记）+ 错误 16（本机缺 onnx/onnxsim/onnxruntime，
   T09 已登记）+ 跳过 20；本票受影响/新增 11 个模块全部通过。新增 `tests/test_perf_engine_models.py`（12 项）：
   规则白名单、QSA 逐 query 和、scan/conv 几何、HiDream 条件与 `net_vmerge_norm` 在 view 前、MoE 守卫、
   `g_ix_sum`/`g_ple_gate`/`g_dsa_ix_sum` 逐 query 几何、表达式节点上限。
6. **HiDream DeepStack 源码引用返修（复核后）**：整网 `net_deepstack_inject` 行的 `source_refs` 改为复用
   `extract.hidream_o1_hf.MODELING` 常量与 `_src()`，指向现存
   `models/HiDream/HiDream-O1-Image/qwen3_vl_transformers.py`，locator `762-774/913-919/938-946`
   （ViT tap / decoder 条件注入 / `_deepstack_process`）。仅此一行“源码/配置位置”文本变化，
   新旧工作簿逐单元格比较**只有 1 个单元格不同**（Generation L111），所有成本/维度/公式与数值不变，
   未重跑原生重算。
7. **接收方独立复核反馈**：Root 已用原生重算独立覆盖四文本模型 16 组 × 2 页（批量/卷积/scan/QSA 跨 cap 边界/
   有效与非法 `N_hit`、串行求和），并确认 DeepSeek 填 `N_hit=6` 两页总计与 T10 完全一致；本报告据其反馈
   修正了“仅表头/冻结窗格变化”的表述（不改产品、不重建）。

## 复用 / 重构 / 新增 / 删除

- **复用**：五模型唯一 `export/operator_workbook.py` 导出链、`hf_workbook_common.py` 的可填写输入/参数/层组/
  整网声明、`scenario_evaluator.py` 白名单编译与三态守卫、各模型同源 `layer_structures.json`/
  `network_overview.json`、`build_hf_decoder.py` 的内容寻址身份。
- **重构**：新增模型无关 `export/perf_engine.py`（默认吞吐/通用文案、`cube`/`vec`/`cube_terms`/`vec_terms`/
  `container` 构造器、符号提取、规则清单、`attach_performance` 全覆盖+符号校验、MoE 命中守卫模板、
  整网外围参数行与 `S_sel` 活公式）；`deepseek_perf_rules.py` 改为复用该基座（输出字节不变）；
  `hf_workbook.py` 的重复绑定校验改为调用基座；`operator_workbook.py` 增加：条件优先的维度/工作量/状态
  渲染（false 分支 0、条件未知仍未知）、层组/串行条件条目（含可选一致性守卫）、串行“未定贡献项数”按实际
  组数生成、成本列标题写清默认与实际吞吐、启用性能区时冻结 C2、`_MAX_NODES=160`（QSA 精确和约 100 节点）。
- **新增**：`export/glm53_perf_rules.py`、`export/glm53flash_perf_rules.py`、`export/qwen38flash_perf_rules.py`、
  `export/hidream_o1_perf_rules.py`；四个适配器接入性能区/整网外围/串行汇总；HiDream 声明式
  `net_deepstack_inject` 整网行与 `vit-tower` 执行条件；`tests/test_perf_engine_models.py`；
  `T11-cube-vec-verify/` 的证据脚本/JSON/36 张渲染图；本报告与交付索引更新。
- **删除**：无。旧候选、旧证据、旧单算子模板、根两项既有删除状态均未改变。

## 限制与未验证

- 性能是用户确认的**归一化粗估基准**（4096 MAC/ms、64 元素/ms），不是设备实测；量级很大是基准本身的结果。
- 不计带宽/并行/融合/通信/精度折扣；chunk/分块内核的算法改写附加工作不另计，只按递推数学核；
  不声称实际 kernel MAC 与递推完全相同。
- `N_hit` 单值是“各层统一命中数”假设，不推断实际路由分布；默认未给定时相关层组与整网总计为未知。
- HiDream 的逐图注意力要求等长参考图（`Kref·S_i=Svis`）；不等长/缺失 → 未知，不平均成单值。
  `net_deepstack_inject` 是声明式整网聚合行（源码事实：ViT tap 8/16/24、decoder 前 3 层），
  不是 network_overview 的原始算子；这是为避免 36/27 层模板重复计而显式声明的唯一新增行。
- 未在 Microsoft Excel GUI 手工编辑；验证引擎是 Artifact Tool + 其原生重算。未做用户阅读验收。
- 全量测试仍有 6 项 diagram 旧布局断言与 16 项 ONNX 缺依赖历史项，不计为本票通过，也未用无关改动掩盖。

## 停止状态

完成本票实现与验证后停止，等待 Codex 复核。GitHub #12 保持 OPEN；未 commit/push；未改
`execution.json`/`tickets.json`/`monitor-state.json`/任务 README 或自动化配置；自动化仍 PAUSED/45 分钟；
未恢复旧交接会话。
