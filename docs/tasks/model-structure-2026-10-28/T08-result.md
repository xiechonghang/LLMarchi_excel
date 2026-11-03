# T08 结果：HiDream-O1-Image 单次神经网络的单 sheet 分层结构

状态：**第三轮返修已完成并通过本票独立验证，停止等待 Codex 复核**（第一轮
`9876cc66…`、第二轮 `1ccc7224…` 均保留；本轮修复 3 项残留 P2 后生成新候选
`5423a22f…`）。
本票只做 T08（GitHub #9）：按官方 `HiDream-ai/HiDream-O1-Image`（HF revision
`0b0901d9…`）与官方 GitHub 实现 commit `2c2d29ff…`，交付**一次网络 forward**
的整网→层组→Block→算子分层 HTML、单 sheet 工作簿、真实语义检查与证据。未
commit/push、未关闭 #9、未进入 T09、未修改协调/自动化文件；未下载权重、未导入/
实例化/运行模型。

## 第二轮复核修复（Codex round-1 的 4 项 P2）

Codex 2026-10-29 第一轮复核暂不验收；第一轮正式候选 `9876cc66…` 原样保留。本轮按
`T08-review/round-1-findings.md` 原位修复，冻结结构后只生成一个新候选并做一次
针对缺陷与共享接缝的验证（未重复昂贵全量测试）。

| 编号 | 复核问题 | 源码/复现依据 | 本轮修复 |
| --- | --- | --- | --- |
| P2-1 | 视觉注意力按所有图总长 `[Nhv,Svis,Svis]` 平方，没有 cu_seqlens/每段 S_i | modeling:240-259（按 `cu_seqlens` 差分切块、逐段 attention 后 concat）、238-247（flash 同分段）；两张各 576 真值 10,616,832，旧结构 21,233,664 | 新增 `v_cu_seqlens`、`v_seg_split`/`v_seg_concat`、逐段张量 `v_q_seg/v_k_seg/v_v_seg/v_ctx_seg`；`v_scores/v_probs` 改 `[Nhv,S_i,S_i]`，`v_ctx` 为 `[Svis,Nhv,Dhv]`；新增每图长度 S_i 输入与 `S_attn = Nhv·Kref·S_i²`（guard `Kref·S_i == Svis`）；契约拒绝总长平方、缺切分/拼接 |
| P2-2 | 时间步 MLP、像素投影、Final、Conv3d/merger 等必要网络任务没有单算子工作簿行 | `hf_workbook_common:250-278` 只降为箭头串；`operator_workbook:1633-1645` 只渲染路径 | 单 sheet 增加第三个分组 `network-ops`（整网一次性任务，28 行）：参数/张量/算子全部来自 `network_overview`，输入/输出/权重 shape 活公式，逐行写执行条件；层循环父节点（36 层/27 blocks）不出行、不与层组重复计成本；仍用唯一导出器 |
| P2-3 | 主线与 DeepStack merger 压成 `norm-mlp-gelu` 黑箱 | modeling:108-121（Norm/view/Linear1/GELU/Linear2；主线 norm 在 view 前 1152、DeepStack norm 在 view 后 4608） | 拆成 `net_vmerge_norm → view → fc1 → gelu → fc2` 与 `net_ds_view → norm → fc1 → gelu → fc2`，新增中间张量（n_vmerge_units/h/act、n_ds_units/normed/h/act）与 norm/线性权重；×3 与 tap 8/16/24 在行说明中标注；L0 只折叠 27 层塔、不重复跑 |
| P2-4 | 输入约束没进入公式/shape/条件（1025 静默 floor、Svis577 静默取 144、K=-1、K0/Sref1024 仍得正常规模） | 原生重算复现的 6 个场景；pipeline:313-317 的 patch reshape、modeling:118-120 的 merge view 需要可整除 | 参数新增可选 `consistency`（白名单布尔表达式），导出器把它编译成 IFERROR 守卫：不满足时该输入与所有依赖量显示“未知（不满足约束：…）”，恢复后重算；Himg/Wimg 必须 `% Ppix == 0`、Svis 必须 `% (Mv·Mv) == 0`、Kref≥0、Sref/Svis/S_i 与 Kref 的存在性/一致性、Ptxt≥1/B≥1；派生式新增可选 `guard`，HTML 场景面板同样在不等长时未知 |

L0 仍折叠 27 层塔（不展开 block）；merger 以 5 个语义步骤展开，未重复展开塔或
block 内部。层组/Block/算子页与单 sheet 工作簿消费同一份结构。

同时修掉两处显示污染（由全局 `display_names` 把最小符号 K/T/P 替换成其它模型字段名）：
HiDream 文案中的参考图数量统一写 `Kref`，patch 维度写 `Cin×Tp×Pv×Pv`；页面不再出现
`num_experts_per_tok`/`past_sequence_length` 等替换。

本轮新增 9 项回归（66→75）：逐段局部 shape、Σ Nhv·S_i²、等长/不等长/缺分段场景、
一致性守卫公式、merger 明细与两个负例（总长平方 score、黑箱 merger）。

Codex 的 2 项判断性 P3（`build_hf_decoder.py:967` 的三个同形闭包重复、
`export/hidream_o1_workbook.py:183` 的 `_parameter_rows` 无调用者）按复核意见
留 T09，本轮未扩大清理。

修复后候选：`1ccc72242a9fb692acb2ebe1e803d184414283aac8ff91a720c6ab62e220d4c8`。
第一轮 `9876cc66…`、第二轮 `1ccc7224…` 保留；本票共删除 15 个中间候选与 3 个
失败记录（第三轮只成功构建最终候选，无新的中间产物；见 `T08-verify/candidate.txt`）。

## 第三轮复核修复（Codex round-2 的 3 项残留 P2）

Codex 2026-10-29 第二轮复核确认外围任务入表与 merger 细化通过，仍有 3 项 P2。本轮按
`T08-review/round-2-findings.md` 只修这三处，不重做已通过项、不扩展 T09。

| 编号 | 复核问题 | 源码/复现依据 | 本轮修复 |
| --- | --- | --- | --- |
| A | `_render_operators` 重建简化 guard，绕过参数完整 domain/consistency；参数区未知但算子实际 shape 仍有效 | round-2 原生重算：Kref=2/Svis=576/S_i=144 时 D197 未知、R221 仍 `[16,144,144]`；Svis=577 时 D196 未知、R216 仍 `[577,3456]` | `_resolve_parameter_values` 按**参数 id** 记录每个参数的单元格与完整守卫（domain+consistency），`_render_operators` 直接使用这份权威映射（不再按符号取最后一个、也不再另建简化 guard）；同名符号消歧保持既有规则 |
| B | `v_cu_seqlens` shape 写成 `[Kref,1]` | modeling:752-760：cumsum 后左端 pad 0，是一维 `[Kref+1]`；Kref=2 应为 `[3]` | 改为 `["Kref+1"]`，备注/源码依据同步；测试与原生重算断言 Kref=0/1/2 → `[1]/[2]/[3]`（v_seg_split 实际输入 shape 列） |
| C | 每图 `S_i` 缺 merge 单元整除，Kref=4/S_i=577/Svis=2308 可通过总长与乘法一致却得到非法局部 shape | modeling:639-652 按每图网格重排；577 无法组成 2×2 merge，不能借别图 patch 补齐 | `S_i` 一致性增加 `S_i % (Mv·Mv) == 0`；派生式 guard 同步；新增场景 `edit-4ref-nondivisible`（Kref=4、577、总 2308）演示未知，恢复 576 后重算 |

新候选：`5423a22f4a8c781c147c3bc80d4005a25417869f29372910d11aca7224663b91`。
前两轮正式候选 `9876cc66…`、`1ccc7224…` 原样保留；第三轮只成功构建这一个候选
（无中间候选需清理）。新增 4 项回归（75→79）：cu_seqlens 一维 shape、S_i merge
整除守卫、非法每图长度的场景未知、算子 Q/R/S 列使用参数完整守卫。

## 接手确认

- 工作区/cwd：`/Users/xiechonghang/Workspace/LLMarchi_excel`。
- HEAD：`947ae78f120279403ec0a4c539e99d3d8869e1d6`（main，与派单一致）；本票未产生 commit。
- 接手快照：`/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T08-baseline-20261029-_rptidcc/manifest.json`
  （546 文件，2026-10-29T12:40:49+08:00 采集；同目录 `files/` 为内容副本）。
  实际 cwd/HEAD/文件数已核对，改动/新增均对照该快照。
- 快照前已存在的 HEAD 脏改动/删除（`D CONTEXT.md`、`D 2026_10_17_workspace_structure_review.md`
  等 21 项 + 73 项未跟踪）与全部接手前候选未动；未 reset/clean、未批量格式化。
- 快照后由协调流程写入/更新的 7 个文件（`docs/handoffs/…`、任务 README/execution/
  monitor/tickets/09.md/executions/T07.json）只读，未编辑；`changed-files.json`
  将其列为 external coordination（差异内容为 Codex 的 T08 派单/状态记录）。
- 接手前候选保持：DeepSeek `c3fd4296…`、GLM-5.3 `abefb101…`、GLM-5.3-Flash 全部
  候选（含 T06 `29f82a3e…`）、Qwen3.8 的 `c0d4ed84…` 与 `c9217122…`。

## 官方来源与配置身份（先锁定，再开发）

| 项 | 值 | 证据 |
| --- | --- | --- |
| 模型仓库 | `HiDream-ai/HiDream-O1-Image` | HF API `/api/models/HiDream-ai/HiDream-O1-Image` |
| 仓库 revision | `0b0901d99f200389e138c61946af1185f5f49a13`（2026-06-22） | 同上；`rules/hidream_o1_hf.py:MODEL_REPOSITORY` |
| 根配置 | `models/HiDream/HiDream-O1-Image/config.json`，1,474 B，SHA256 `5cd45286…8f3661` | 与官方 revision 的 raw config 逐字节相同 |
| 模型卡 | `models/HiDream/HiDream-O1-Image/README.md`，23,917 B，SHA256 `901f12f2…e0e7b7` | 与官方 raw README 逐字节相同 |
| model_type / architectures | `qwen3_vl` / `["Qwen3VLForConditionalGeneration"]`；`text_config.model_type=qwen3_vl_text` | 根配置；`load_hf_model_values` 双条件校验 |
| 实现（官方代码仓库） | GitHub `HiDream-ai/HiDream-O1-Image` commit `2c2d29ff729e48f33e41f49edfdbd81d5ac103b4`（2026-06-22） | GitHub API `/commits/main`；`source_revision` |
| 已分析源码 | `models/qwen3_vl_transformers.py`（101,540 B）、`pipeline.py`（19,560 B）、`utils.py`（14,733 B）、`inference.py`（5,890 B） | 4 个小文件的 SHA256 见下 |
| checkpoint 保存版本 | config `transformers_version="4.57.0.dev0"` | 记录在 `data/effective_config.json` |
| 权重 | **未下载/未读取任何权重分片**（仓库 8 个 shard，合计约 35.2 GB 全部未取） | 只取 config、README 与 4 个源码文件 |

已分析输入指纹（SHA256）：

```
config.json                  5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661
README.md                    901f12f248a2e43a7ac6092d2a6de9f68f444c83e8ba4a53d55eb3069240e7b7
qwen3_vl_transformers.py     5614a339654a77e1af7c455a07e3665bfcf60da476b37097aa123b3c051ab17b
pipeline.py                  68156223cfa6cf86e52776cf8480cd5446d311dd3313a8104adc0355bf0460a6
utils.py                     32faef1e2b2fe4c5e75fc12dbefb89d708dfccb1cd390a2a59d2f9011c319a7e
inference.py                 df1280dc648193fa007780b4618a89cda3004758ba99d0e81dfbe28f575b4031
```

**身份辨伪**：这是 **HiDream-O1-Image（8B，Pixel-level Unified Transformer / UiT）**，
不是 HiDream-I1（13.5B + 17B 双流 DiT）、不是 `HiDream-O1-Image-Dev`、也不是任何
DiT+VAE 或双流 MMDiT。模型卡明确 “natively unified … without external VAEs or
disjoint text encoders”；实现上它复用 Qwen3-VL 的 decoder 结构，但生成路径**不是**
自回归 next-token（`vinputs` 分派到 `_forward_generation`，返回 `x_pred`）。
config 记录的 transformers 版本（4.57.0.dev0）只是保存版本；本票以官方仓库
**vendored 的 `models/qwen3_vl_transformers.py` 实际代码**为唯一实现版本，没有用
已安装的 transformers 包替代。

## 关键静态解释（每条有源码行号证据，见 `rules/hidream_o1_hf.py`）

- **单次 forward 的入口与终点**：`Qwen3VLForConditionalGeneration.forward` →
  `Qwen3VLModel.forward`；检测到 `vinputs` 即转 `_forward_generation`
  （modeling:1650-1660），返回 `x_pred`（每 token 32×32×3），**不调用 lm_head**
  （modeling:1906-1912）。`lm_head` 权重仍在模型里（自回归分支用），不属于本次
  生成 forward。
- **文本主干**：36 层 `Qwen3VLTextDecoderLayer`，hidden 4096、32 Q 头 / 8 KV 头、
  head_dim 128、SwiGLU 12288、RMSNorm eps 1e-6、逐头 q_norm/k_norm、MRoPE
  interleaved `[24,20,20]`、theta 5e6、vocab 151936、attention_bias=false
  （modeling:293-539/787-936；config text_config）。
- **像素 patch 嵌入**：`BottleneckPatchEmbed` = Linear(32·32·3=3072 → 1024) →
  Linear(1024 → 4096)；pca_dim = hidden//4 = 1024（modeling:947-963/1039-1045）。
- **时间步嵌入**：正弦 256 维 → Linear(256→4096) → SiLU → Linear(4096→4096)
  （modeling:983-1022）；`t_embedder2=None`，`FinalLayer.adaln_input` 未使用 ——
  **没有 adaLN / 时间步调制 MLP**（modeling:977-979/1047-1049）。
- **时间步如何进入网络**：`tms_token_id=151673` 硬编码；`torch.where(input_ids ==
  tms, t_emb, inputs_embeds)` 把它**写进 1 个 token 位置**（modeling:1050/1520-1523），
  不是加到每个 token 上。
- **序列构造**：文本 ids = chat template + boi + 1×tms（pipeline:31-78）；目标图像
  patch 与参考图 patch 都经同一个 `x_embedder` 后拼在文本之后：
  `[文本嵌入; x_embedder(vinputs)]`（modeling:1525-1532）。`token_types`
  0=AR 文本、1=目标生成、2=参考图 patch、3=tms；`vinput_mask` 只覆盖生成 token
  （pipeline:62-77/294-302）。
- **注意力语义（同一层内的行级差异）**：AR 文本行只对 AR 文本做因果注意力；生成
  token 行对全部 token 双向。推理走 flash 两趟：第 1 趟在 AR 子集上 causal、第 2 趟
  在全序列上 non-causal，再 `out_full[:, idx_ar] = out_ar`（modeling:1331-1345）；
  `use_flash_attn=False` 时用等价的 4D mask（生成行置 0、其余上三角 -inf，
  modeling:1577-1589）。文本在前、生成 token 在后的布局使两种实现语义一致。
- **K>0 时文本部分还含条件图 placeholder**：processor 会把每张条件图的
  `<|image_pad|>` 占位 token 写进文本 ids（pipeline:260-266），它们计入 `Ptxt`、
  属 AR 行（token_types=0），随后被 `masked_scatter` 替换成视觉塔嵌入。t2i 没有这些
  token；workbook 的 `prompt_tokens` 应按实际 tokenizer 结果填写（含这些占位）。
- **输入口径与一致性**：`Himg`/`Wimg` 是**实际生效网络尺寸**，必须能被像素 patch
  边长整除（`% 32 == 0`），不能把原始请求尺寸静默 floor；视觉 merge 要求
  `Svis % 4 == 0`；`Kref`/`Sref`/`Svis`/`S_i` 必须互相一致（Kref=0 → 三者=0；
  Kref>0 → 三者>0 且 `Kref×S_i=Svis`）。不满足时参数、shape、算子区与场景面板都
  显示“未知（不满足约束：…）”，恢复后重算——约束进入公式而不是只写在提示里。
- **Merger 粒度**：主线 merger 是 `LayerNorm(1152) → view(-1,4608) → Linear(4608→4608)
  → GELU → Linear(4608→4096)`；DeepStack 的 3 个同构 merger 是
  `view → LayerNorm(4608) → Linear → GELU → Linear`（postshuffle，norm 在 view 后），
  分别读取 block 8/16/24 的 tap（modeling:108-121/770-774）。
- **位置**：`get_rope_index_fix_point` 把目标图像 patch 的位置钉在 **4096 起**的
  h/w 网格（与文本长度无关），vision_start token 不计入位置；参考图 patch 的位置
  继续累加（utils:246-349；pipeline:48-60/285-290）。
- **输出选择**：`final_layer2` 作用于全部 token，再按 `vinput_mask` 取生成 token，
  参考图任务取前 `tgt_image_len` 个（modeling:1612-1613；pipeline:350-357）。
- **视觉塔（条件执行）**：27 个 `Qwen3VLVisionBlock`（hidden 1152、16 头、
  head_dim 72、MLP 4304、LayerNorm、2D rotary、patch 16×16、temporal 2、
  spatial merge 2）；注意力按 `cu_seqlens` **逐图切段**、段内非因果
  `[Nhv,S_i,S_i]`、再 concat（modeling:238-259），总元素是 `Σ_i Nhv·S_i²` 而不是
  `Nhv·(Σ_i S_i)²`。`deepstack_visual_indexes=[8,16,24]` 在第 8/16/24 个 block 后
  各接一个 DeepStack merger，主线再有 1 个 merger（modeling:589-778）。
  该塔**只在参考图/编辑/IP 任务（K>0）执行**：t2i 路径 `pixel_values=None` 且
  inference 在 `torch.no_grad` 下，不跑 dummy 视觉（modeling:1440-1479）。
- **DeepStack 注入**：`hidden_states[visual_pos_masks] += visual_embeds`，只改条件图
  占位行；3 个特征分别注入 decoder 的 layer 0/1/2（modeling:913-919/938-945）。
- **缓存**：生成 forward **不使用 past_key_values**（`use_cache=False`，
  modeling:1603；`_run_decoder_flash` 直接循环 layers，modeling:1368-1378）。
  单次 forward 没有 KV cache 读写，本结构没有 cache 张量。
- **外部边界（不分析）**：50 步（full）/28 步（dev）采样循环、scheduler
  （FlowUniPCMultistep / FlowMatchEuler / FlashFlowMatch）、CFG 的 cond/uncond 两次
  调用与 `v` 组合、噪声注入与 `v=(x_pred-z)/sigma`、最终像素反拼（pipeline:313-435；
  inference.py:79-119）。它们不进入网络结构、不乘步数、不算完整出图耗时。

**明确未分析**：loss/反向/FSDP/梯度检查点/训练期 dummy 视觉前向、prompt agent
（Gemma-4-31B）、layout/skeleton 条件图预处理、bfloat16 数值等价、设备 kernel、
flash-attn 版本差异、torch.compile。

## 场景（同一参数化结构，按本模型输入规模绑定）

结构 shape 保留符号；场景只绑定取值。场景轴不是 seqlen/kvcachelen，而是
`HIDREAM_BIND_KEYS = (Himg, Wimg, Ptxt, Kref, Sref, Svis)`。

| 场景 | 图像 | prompt | K | 目标 patch | 生成 token | 序列 | AR 行 | 生成侧行 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t2i-1024 | 1024×1024 | 128 | 0 | 1024 | 1024 | 1152 | 127 | 1025 |
| t2i-2048 | 2048×2048 | 128 | 0 | 4096 | 4096 | 4224 | 127 | 4097 |
| edit-1ref | 1024×1024 | 128 | 1 | 1024 | 未知（Sref 未给定） | 未知 | 127 | 未知 |
| t2i-symbolic | 未给定 | 未给定 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 |

`N_vis = Svis // (Mv·Mv)`（DeepStack 特征行数）：t2i 为 0；edit 未给条件图尺寸时
未知。未知项如实标注原因，不默认 0；结构图/工作表共用同一份派生式。

## 交付入口与真实产物

```bash
PY=/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
cd /Users/xiechonghang/Workspace/LLMarchi_excel
PYTHONPATH=structure_reader/src "$PY" structure_reader/scripts/read_model_structure.py \
  build-hf --model hidream_o1_image_hf --all-variants --json
```

- **最终候选（第三轮）**：`5423a22f4a8c781c147c3bc80d4005a25417869f29372910d11aca7224663b91`
  （重复执行同一命令 `reused: true`，不新增候选）。
- **真实 HTML**：[viewer/structure.html](../../../structure_reader/outputs/hidream_o1_image_hf/5423a22f4a8c781c147c3bc80d4005a25417869f29372910d11aca7224663b91/viewer/structure.html)
  （666,679 B；SHA256 `f6f36c86…4a880e051`）
- **真实 Excel**：[tables/decoder_variants.xlsx](../../../structure_reader/outputs/hidream_o1_image_hf/5423a22f4a8c781c147c3bc80d4005a25417869f29372910d11aca7224663b91/tables/decoder_variants.xlsx)
  （52,371 B；SHA256 `078b0bc3…e9097e2`）——**单 sheet `Generation`**，3 个分组
  （network-ops 28 + uit-decoder 30 + vit-tower 19 = 77 行）
- 同候选：`data/layer_structures.json`（151,456 B；`06de5e93…57b8e2`）、
  `data/network_overview.json`（74,354 B；`2a553b93…2fb75190`）、
  `data/effective_config.json`（13,185 B；`6b06eb8e…4d38700`）、
  `reports/source_manifest.json`（`367ffddb…52cf0da`）、`run_manifest.json`。
- 前两轮正式候选 `9876cc66…`（第一轮）与 `1ccc7224…`（第二轮）原样保留，
  不作为本轮修复后的版本。
- **清理的本票中间候选**（仅本票、仅 HiDream 目录，已精确删除并记录）：
  第一轮 `67e6f0ce…`（首个 draft，清掉两处跨模型比较文字后重建）、`be310b79…`、
  `aa5dad6a…`、`7bf9f966…`、`7ecf304a…`、`fe3a39b9…`、`ab2c6b52…`、`0a714a59…`；
  第二轮 `b340c236…`、`4011f868…`、`aa55475e…`、`384d4c47…`、`f0dcf44f…`、
  `778301f2…`、`8456b321…`；另删除本票失败记录 3 个（首轮 NameError 与第二轮的
  检查/公式失败，均未发布产物）。接手前候选均未动。

## 已实现

### 1. 分层数据（同一份参数化结构）

- `extract/hidream_o1_hf.py`：两个真实重复层组各一份 Block 结构，shape 保留符号。
  - `uit-decoder`（36 层，0..35）：30 算子 / 49 张量 / 25 参数 / 6 派生式。
    input_layernorm → q/k/v 三投影（q 4096、k/v 1024，无 bias）→ 逐头 q_norm/k_norm
    → MRoPE（只作用 q/k）→ flash 两趟（AR 子集 causal、全序列双向）+ index_copy →
    拼头 + o_proj → 标量残差 → post_attention_layernorm → SwiGLU（gate/up 12288、
    SiLU、乘、down）→ 标量残差 → 条件 DeepStack 注入（K>0 且 layer 0..2）。
  - `vit-tower`（27 blocks，0..26，条件执行）：19 算子 / 34 张量 / 17 参数 /
    4 派生式。LayerNorm → 合并 qkv（3456，bias）→ 拆分 → 2D rotary（q/k）→
    **按 cu_seqlens 逐图切段**（`v_seg_split`，第 i 段 `[Nhv,S_i,Dhv]`）→ 段内非因果
    QK/softmax/AV（`[Nhv,S_i,S_i]`）→ 逐段 concat（`v_seg_concat`）→ 拼头 + proj →
    残差 → LayerNorm → GELU-tanh MLP（4304）→ 残差 → block 8/16/24 的 DeepStack tap。
  - 两个索引空间**各自从 0 开始**，不用 `layer_index` 反查（`build_layer_documents`
    在只给 layer_index 时具名拒绝），容器通过新增的
    `build_structures_container(build_variant_fn=…)` 按 group 构建。
- `extract/hidream_o1_network.py`：单次 forward 整网（30 算子 / 60 张量 / 44 参数 /
  14 派生式）：文本 embedding → 时间步正弦 + 2 层 MLP → tms 位置替换；
  可选视觉塔（patch Conv3d → 位置插值 → 2D rotary → 27 blocks → 主线 merger
  （Norm(1152) → view → fc1 → GELU → fc2）+ 3 个 DeepStack merger（view → Norm(4608)
  → fc1 → GELU → fc2）→ masked_scatter）；像素 patch 两段低秩投影；
  文本 + 生成 token 拼接；MRoPE；AR/生成行划分；36 层循环；Final RMSNorm；
  final_layer2；按生成 token 取 `x_pred`。

### 2. 分层浏览页（复用同一渲染核心）

- `viewer/structure.html`：7 个层级 section（整网 + 2×(层组/Block/算子)）、27 个 SVG、
  631 个唯一 id、普通锚点全部闭合；默认只显示当前层级，面包屑/链接逐级进入与返回；
  场景按钮切换 7 个预设（文生图 1024/2048、单参考图、两张等长、两张不等长、
  四张每图 577 非 merge 单元、符号）；
  源码快照、参数/张量/图/算子联查沿用既有锚点体系。
- 两个层组分别显示真实索引与执行条件：`uit-decoder` 36 层总是执行、`vit-tower`
  27 blocks 仅 K>0 执行；网络层说明“视觉塔是条件分支，不占 36 层的位置”。
- 模型 profile（`render/hf_model_profile.py`）只替换展示文案/模块规则/布局/图例/
  源码命名表，不复制页面装配；新文案字段默认值与既有页面逐字节一致（见“跨模型
  不回退”）。图例只含本模型机制（两趟注意力、条件 DeepStack、视觉塔条件执行），
  不含 KDA/mHC/Sinkhorn/GDN 等其它模型机制。
- 页面 header 的“shape 保留符号”说明也改为 profile 驱动（默认字符串不变），
  HiDream 显示“图像高/宽、prompt token 数、参考图 patch 数、条件图 token 数；
  本网络没有 KV cache”。

### 3. 同源单 sheet 工作簿

- `tables/decoder_variants.xlsx`：**单 sheet `Generation`**，3 个分组、**77 个算子行**
  （network-ops 28 + uit-decoder 30 + vit-tower 19），全部来自与 HTML 同一份结构：
  network-ops 是整网一次性任务组（时间步 MLP、像素投影、视觉塔入口/merger、Final），
  层循环父节点（36 层/27 blocks）不重复成行；两组真实层组各一份 Block 模板。
- 顶部 11 项可填写输入：`image_height`、`image_width`、`prompt_tokens`、
  `ref_images`、`ref_patch_tokens`、`cond_vision_tokens`、`cond_image_tokens`、
  `batch_size`、`compute_precision`、`weight_storage_bits`、`activation_storage_bits`。
- 派生参数是引用本页输入的活公式（带 `IF`/`ISNUMBER`/`INT`/`IFERROR` 守卫）：
  `S_pix = (Himg // Ppix) * (Wimg // Ppix)`、`S_gen = S_pix + Sref`、
  `S_tot = Ptxt + S_gen`、`N_ar = Ptxt - 1`、`N_gt = S_gen + 1`、
  `N_vis = Svis // (Mv * Mv)`、`H_q = Nh * Dh`、`H_kv = Nkv * Dh`、
  `S_vis_m = Svis // (Mv * Mv)`、`H_v3 = 3 * Hv`、`D_hv = Hv // Nhv`、
  `H_v4 = Hv * (Mv * Mv)`、`V_pin = Cin * Tp * Pv * Pv`、
  `P_in = Ppix * Ppix * Cin`、`P_ca = H // 4`。
- 参数带 domain 守卫（Himg/Wimg ≥ 32、Ptxt ≥ 1、B ≥ 1、Kref ≥ 0）与 **consistency
  一致性守卫**：Himg/Wimg 必须被 Ppix 整除、Svis 必须被 merge 单元整除、
  Kref/Sref/Svis/S_i 必须互相一致（Kref=0 → Sref=Svis=S_i=0；Kref>0 → 三者 >0、
  **S_i 必须被 merge 单元整除（S_i % 4 = 0）** 且 Kref×S_i=Svis）。不满足时该输入与所有依赖量显示“未知（不满足约束：…）”，
  恢复合法值后由 Excel 重算；网络一次性任务行的输入/输出/权重 shape 同样活引用。
- 成本列统一“待提供”，无猜测耗时、无成本 SUM；整网摘要明确“不把 36 层与 27 blocks
  相加成 63 层”（新增可选 `count_note`，旧模型不传时工作簿字节不变）。

### 4. 检查（发现真实错误，不照实现镜像）

`checks/hidream_o1_hf_contracts.py`：
- 身份/单阶段声明/骨架/数据流闭合；拒绝 Prefill/Decode 两阶段投影与 cache 张量；
- decoder：GQA 32/8、逐头 q/k RMSNorm、V 无 norm/RoPE、MRoPE 只作用 q/k、
  两趟注意力 + index_copy（AR score 轴必须是 `Nar`、全序列是 `Stot`）、两个 RMSNorm
  站点、两个标量残差、SwiGLU 12288、DeepStack 条件（K>0 且 layer 0..2）、
  禁 MoE/mHC/Sinkhorn/KDA/线性注意力/adaLN/VAE/文本编码器；
- vision：LayerNorm（不是 RMSNorm）、合并 qkv + bias、2D rotary、非因果分图注意力、
  GELU-tanh 4304、tap 8/16/24、无 cache；
- 整网：终点 `x_pred` 不是 logits、无 lm_head/采样调度/VAE/独立文本编码器、
  36 层真实顺序、视觉塔条件执行、tms id/deepstack indexes 固定、位置/掩码输入存在。

## 复用 / 重构 / 删除 / 新增

### 保留复用

- `build_hf_decoder.py` 的资料核验/身份/内容寻址发布/失败隔离；`cli._materialize`。
- `extract/hf_scenario.py` 的场景投影与三态求值；`scenario_evaluator.py` 白名单与
  Excel 编译。
- `render/diagram.py` 的真实张量引用建图/SVG/锚点；`render/detail_render.py` 的算子卡/
  张量/参数联查；`render/hf_structure_page.py` 的层级/路由/场景条。
- `export/operator_workbook.py` 唯一 Workbook 创建/sheet/保存/稳定 ZIP；
  `export/hf_workbook_common.py` 的输入/参数/组/整网声明。
- `checks/hf_contract_core.py` 的通用骨架/数据流/输入身份检查。

### 原位重构（迁移调用后无重复实现）

| 文件 | 改动 |
| --- | --- |
| `build_hf_decoder.py` | `_Adapter` 注册表新增 `hidream_o1_image_hf`：共同编排、来源/工具 SHA、检查、渲染、导出仍只一处；单层/ONNX 显式拒绝（文案补 HiDream）。 |
| `extract/hf_scenario.py` | `evaluate_derivations`/`structure_document` 新增可选 `bind_keys`（场景轴）与 `default_scenario`；派生式新增可选 `guard`（白名单布尔表达式，为 False 时该派生量未知）；既有文本模型不传时输出不变。 |
| `extract/hf_layer_structures.py` | 新增可选 `build_variant_fn(workspace, group, model_key)`：同一模型有多个独立索引空间的层组时按 group 构建；不传时行为不变。 |
| `render/hf_structure_page.py` | 把网络标题/说明、Block 标题、层组类型标签、场景条/面板文案、header shape 说明改为 profile 驱动；默认字符串与既有页面逐字节一致。 |
| `render/hf_model_profile.py` | `PageProfile` 追加带默认值的展示字段（文案/标题/场景说明）；新增 `HIDREAM_O1_PAGE_PROFILE`（模块规则、总览布局/frames、讲解、源码表、图例、执行条件）。DeepSeek/GLM-5.3/Flash/Qwen profile 不改。 |
| `export/hf_workbook_common.py` | `network_section` 新增可选 `count_note`；`parameter_rows` 传递可选 `consistency`；不传时不变。 |
| `export/operator_workbook.py` | 整网摘要支持可选 `count_note`；参数新增可选 `consistency`（编译成 IFERROR 守卫，不满足时该参数与依赖量未知，非法文本/空白不产生 `#VALUE!`）；不传时字节不变。 |
| `cli.py` | `--model` 帮助列出 HiDream 模型键。 |
| `structure_reader/README.md`、`docs/hf_reading_variants_contract_v1.md` | 增加 HiDream 入口、身份/边界与 T08 增量说明。 |

### 新增

| 文件 | 职责 |
| --- | --- |
| `rules/hidream_o1_hf.py` | 官方身份/revision、6 项源码指纹、单次 forward 边界、配置静态解释（文本/视觉/像素 patch/时间步/位置/tms/attention 语义）、两组归属、参数来源记录、4 个场景。 |
| `extract/hidream_o1_hf.py` | 两个层组的参数化 Block 结构（decoder 30 算子、vision 19 算子，含 cu_seqlens 逐段切分/拼接与局部 shape）与单次 forward 层文档（`hidream_o1_layer_document.v1`，不投影两阶段）。 |
| `extract/hidream_o1_network.py` | 单次 forward 整网总览（30 算子；merger 细化为 Norm/view/Linear/GELU/Linear）与 36 层真实顺序/两组归属/外部边界。 |
| `checks/hidream_o1_hf_contracts.py` | HiDream 语义检查：身份/单阶段/两趟注意力/QK-norm/MRoPE/SwiGLU/DeepStack/视觉塔/整网终点/禁词；拒绝 cache、MoE、adaLN、VAE、lm_head、两阶段投影。 |
| `export/hidream_o1_workbook.py` | HiDream 薄适配：**单 sheet 3 组**（network-ops 一次性任务 + 两组真实层组）、输入映射（含 S_i）、15 个派生式映射、循环父节点排除与真实归属校验、成本待提供。 |
| `render/hidream_o1_narrative.py` | 模块/总览中文讲解（decoder 与 vision 分支）。 |
| `render/hidream_o1_source_labels.py` | 算子源码表达式/张量短名/行号（modeling + pipeline）。 |
| `tests/test_hidream_o1_structure.py` | 79 项回归：身份/层组/decoder/vision（逐段注意力、cu_seqlens 一维、S_i merge 整除）/整网（merger 明细）/场景（等长/不等长/非整倍）/契约负例/单 sheet 3 组/HTML/跨模型隔离。 |
| `models/HiDream/HiDream-O1-Image/{config.json,README.md,qwen3_vl_transformers.py,pipeline.py,utils.py,inference.py}` | 官方 revision 的小文件来源（config 1,474 B + README 23,917 B + 4 个源码，共约 167 KB），不含权重。 |
| `docs/tasks/model-structure-2026-10-28/T08-verify/` | 本票验证脚本与证据（见下）。 |

### 删除 / 退役

- 本票未删除产品文件；仅删除本票 15 个中间候选（两轮，含 1 个 draft）与本票 3 个失败
  记录（见上）。DeepSeek
  单层/四组 ONNX、GLM-5.3 分层页、GLM-5.3-Flash 分层页、Qwen3.8 分层页与全部
  接手前候选保留。

## 实际验证证据

### 相关测试（本票环境）

| 测试 | 数量 | 结果 |
| --- | --- | --- |
| `tests/test_hidream_o1_structure.py`（本票新增） | 79 | OK |
| `tests/test_qwen38flash_structure.py` | 49 | OK |
| `tests/test_glm53flash_structure.py` | 62 | OK |
| `tests/test_glm53_structure.py` | 30 | OK |
| `tests/test_hf_decoder_reading.py` | 61 | OK |
| `tests/test_hf_layer_levels.py` | 41 | OK |
| `tests/test_hf_network_overview.py` | 21 | OK |
| `tests/test_operator_workbook_v2.py` | 24 | OK |
| `tests/hf/test_operator_workbook.py`（v1 兼容） | 57 | OK |
| `tests/test_hf_build_missing_xlsx.py` | 2 | OK |
| `tests/test_scenario_evaluator.py` | 45 | OK |
| `tests/test_m2_guard.py` | 2 | OK |
| `tests/test_page_topology_r2b.py` / `r2a` / `r1` | 30 / 27 / 17 | OK |
| `tests/test_page_js_execution.py` | 2 | OK |
| `tests/test_model_registry.py` | 15 | OK |

合计 **564 项**（79 项本票新增 + 485 项既有直接相关；明细见
[`tests.json`](T08-verify/tests.json)）。未跑的 `test_hf_decoder_diagram` 在 T05
接手时已有 6 项旧布局断言失败，本票未改该历史测试；ONNX 测试未计为通过（本环境
缺 onnx 依赖，且 HiDream 显式拒绝 ONNX）。

### 最终候选独立验证

[`final-independent-artifacts.json`](T08-verify/final-independent-artifacts.json)：
`status=passed`，对象是落盘候选 `1ccc7224…`：

- `run_manifest` 7 个交付 SHA、6 项输入 SHA、26 项工具 SHA 全部复算一致；
  `analysis_id` 与候选目录/identity 闭合。
- 数据：两组 36/27 恰好覆盖各自索引空间；decoder 30 算子/49 张量/25 参数、
  vision 19/34/17、network 30/60/44；场景独立期望 t2i-1024 = 1024/1024/1152/127/1025、
  t2i-2048 = 4096/4096/4224/127/4097；两张等长（576+576）逐段注意力总数
  10,616,832、两张不等长（1200）与每图 577（非 merge 单元）时为未知；edit 的
  S_gen/S_tot 未知、符号场景的长度派生全部未知、`N_vis=0`；cu_seqlens 为一维
  `[Kref+1]`；effective_config 与根 config.json 逐字段一致。
- 契约负例：删除两趟合并、加 KV cache、AR score 改全长、混入 MoE、视觉塔标成总是
  执行、文档改 Prefill/Decode 两阶段、视觉 score 改成总长平方、黑箱 merger ——
  全部被具名拒绝。
- HTML：666,679 B、631 个 id 无重复、普通锚点全部闭合、7 个 section、27 个 SVG；
  页面散文无 KDA/mHC/Sinkhorn/Gated DeltaNet/Qwen Sparse Attention/GLM/DeepSeek/
  Qwen3.8 等其它模型机制（源码快照中的官方注释不计为页面事实）。
- 工作簿：单页 `Generation`、3 组 77 行（28+30+19）、11 项可填写输入、派生参数为
  活公式、循环父节点不出行、成本“待提供”、无成本 SUM；v_qk 的实际输出 shape 使用
  S_i 的一致性守卫（不是只查输入非负整数）。

### 现成工作簿公式重算（同一文件，不重新导出）

[`formula-recalc-check.json`](T08-verify/formula-recalc-check.json)：`status=passed`，
对象是同一份落盘 `tables/decoder_variants.xlsx`（SHA256 `79982d42…a8dfccba2`）的
临时副本，用**安装版原生 LibreOffice kit** 的 `recalculate` CLI 改单元格后重算
（`backend=native`；不是修改 spec 重新导出、不是重新生成工作簿）。24 例：

- 原场景：`S_pix/S_gen/S_tot/N_ar/N_gt/N_vis=1024/1024/1152/127/1025/0`；
  `H_q/H_kv=4096/1024`；vision `S_vis_m=0`、`H_v3=3456`。
- 改图像 2048×1024、prompt 256、B=2 → `2048/2048/2304/255/2049`。
- 保持 1024×1024、只改 prompt=256 与 B=2 时，算子卡的实际 shape 随输入重算为
  `[2,1280,4096]`（256 + 1024 个 token），不是重新导出。
- 参考图任务（K=1、Sref=1024、Svis=576）→ `S_gen=2048`、`S_tot=2176`、
  `N_gt=2049`、`N_vis=144`、vision `S_vis_m=144`。
- 改局部架构参数 `Nkv=4` → `H_kv=512`，`H_q` 仍 4096（公式随本表参数重算）。
- 非法（image_height=`abc`）、空白（image_width 空）、prompt=0（domain 守卫）→
  相关量显示“未知（原因）”，其余不受污染；恢复后回到基线值。
- 全表扫描无 `#VALUE!/#REF!/#DIV/0!/#NAME?/#N/A` 等 Excel 错误值（一致性守卫已用
  IFERROR 兜住文本/空白参与比较的情况）。

### 页面渲染与真实点击证据（headless Chrome，非用户验收）

- [整网页](T08-verify/t08-network.png)、[uit-decoder 层组](T08-verify/t08-uit-group.png)、
  [uit-decoder Block](T08-verify/t08-uit-block.png)、
  [AR 注意力算子层](T08-verify/t08-uit-attn-ar-ops.png)、
  [vision Block](T08-verify/t08-vit-block.png)。
- [`browser-navigation-check.json`](T08-verify/browser-navigation-check.json)：
  headless Chrome + 同源 iframe 做**真实 DOM click**，24 步：整网 → uit-decoder
  层组 → Block → 模块 d_attn_ar → 张量 d_ar_index → 参数 Nar → 打开源码面板 →
  返回 Block → 返回整网 → 场景切到 edit-1ref（面板出现未知项）→ 切回 t2i-1024 →
  vit-tower Block → 模块 v_attn_core → 张量 v_scores → 参数 Dhv → 逐图分段模块
  v_attn_seg → 张量 v_q_seg → 参数 S_i → 场景切到 edit-2ref-unequal（面板未知）→
  切到 edit-2ref-equal（显示 10,616,832）→ 返回层组；
  24 项检查全部 `true`，无错误、无重复 id、无缺目标锚点。
- 未做人工鼠标/键盘/窄屏阅读，也未做真实用户阅读认可。

### 跨模型不回退

[`cross-model-regression.json`](T08-verify/cross-model-regression.json)：用本票代码在
内存重建四个既有模型（不发布新候选、不改旧候选），与既有最终候选比较：

| 模型 | 结构 JSON/网络/工作簿 | HTML |
| --- | --- | --- |
| DeepSeek（T04 `c3fd4296…`） | 完全相同 | 归一化候选 id 与 T05/T06/T07 已验收的共享 CSS 注释后相同 |
| GLM-5.3（T05 `abefb101…`） | 完全相同 | 完全相同 |
| GLM-5.3-Flash（T06 `29f82a3e…`） | 完全相同 | 完全相同 |
| Qwen3.8-Flash-Next（T07 `c9217122…`） | 完全相同 | 完全相同 |

说明：本票新增 `bind_keys`/`build_variant_fn`/profile 文案字段/`count_note`/
派生式 `guard`/参数 `consistency`/按参数 id 的权威守卫传递都是向后兼容的改动；`display_names.py` 未改
（未新增全局替换），HiDream 的 `Hq/Hkv` 等新符号不与既有模型冲突；HiDream 文案避开
会被其它模型显示名展开的最小符号（K/T/P 在既有表里分别是专家数/序列长度/历史长度），
页面不出现 `num_experts_per_tok`/`past_sequence_length` 等替换。HiDream 同进程连跑
两次逐字节相同。

### 改动清单

[`changed-files.json`](T08-verify/changed-files.json) 与
[`incremental.patch`](T08-verify/incremental.patch)：相对 T08 快照（546 文件）
修改 17、删除 0、新增 121：

- 产品增量：修改 10（`build_hf_decoder.py`、`cli.py`、`hf_scenario.py`、
  `hf_layer_structures.py`、`hf_structure_page.py`、`hf_model_profile.py`、
  `operator_workbook.py`、`hf_workbook_common.py`、`structure_reader/README.md`、
  `docs/hf_reading_variants_contract_v1.md`）+ 新增 8（7 个模块 + 1 个测试）。
- 证据：21 项（`T08-result.md` + `T08-verify/` 的 7 个脚本、6 个 JSON、1 个补丁、
  `candidate.txt`、5 张截图）；`changed-files.json`/`incremental.patch` 会把自己上一次
  生成的内容列进新增清单（文件内有 self_reference_note）。
- 协调/复核文件 22 项（快照后由 Codex 写入/更新：派单、状态记录、两轮 T08-review
  证据）只读，未编辑；77 项 `models/` 输入中 71 项为接手前已存在，6 项为本票新增的官方
  小文件。

## 未验证 / 明确限制

- 用户性能公式与 XPU 能力仍未提供：无算子耗时/组小计/整网串行汇总；成本列“待提供”，
  不猜、不记 0、不做父子重复计数。
- HiDream-O1-Image **只提供分层入口**（`--all-variants`）；单层长页未实现，缺该参数时
  具名拒绝。未实现可选 ONNX，显式 `--onnx` 具名拒绝（不虚构、不静默忽略）。
- 只覆盖**一次网络 forward**：50/28 步采样循环、CFG 两次调用、scheduler 更新、噪声
  注入、像素反拼与完整出图耗时都明确不在范围；不乘步数。
- t2i 的视觉塔不执行，但其 27-block 结构仍作为条件层组展示；`vit-tower` 与
  `uit-decoder` 是独立索引空间，不能相加成 63 层。
- 数据相关量（参考图 patch 数、条件图 token 数/每图长度、哪些行是条件图占位）保留
  符号/输入，不从权重或 prompt 内容推断；workbook 中这些量按用户填写值代入。
- 单值 S_i 只能表达“各图等长”；两张不等长时无法用一个 S_i 表示各段长度，
  逐段/总注意力规模如实未知（结构保留 `S_i` 符号与逐段循环，未枚举所有图）。
- 未做 Microsoft Excel GUI 打开验证；重算使用安装版原生 LibreOffice kit
  （`backend=native`，日志提示缺少 Calibri 字体，不影响公式求值），结果语义可核对，
  但不等于 Excel GUI 验收。未做模型数值等价、未运行/实例化模型、未加载权重。
- 静态工程、公式引擎重算与 headless 渲染通过不代表用户阅读认可；需 Codex/用户复核。
- 可选 Archify：本票**未实际调用**。模型结构稳定后，L0 总览与局部机制图已由同一份
  结构 JSON 经既有 renderer 生成；没有为出图引入平行 viewer、没有另建一套模型结构、
  没有为出图 commit。图形检查不代替语义/公式验收。

## 保护与停止

- 未 reset/clean、未批量格式化、未恢复根两项删除、未改接手前未提交改动；相对 T08
  快照的改动见 `changed-files.json` 与 `incremental.patch`。
- 接手前 DeepSeek `c3fd4296…`、GLM-5.3 `abefb101…`、GLM-5.3-Flash 全部候选及
  Qwen `c0d4ed84…`/`c9217122…` 未动；本票保留第一轮正式候选 `9876cc66…` 与最终候选
  `1ccc7224…`，精确清理两轮共 15 个中间候选与 3 个失败记录。
- 协调/自动化文件只读，未编辑；自动化保持派单时状态（暂停），本票未修改、未恢复。
- **完成即停止，等待 Codex 复核**：不 commit/push、不关闭 #9、不进入 T09。
