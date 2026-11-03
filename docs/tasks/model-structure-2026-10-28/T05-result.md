# T05 结果：GLM-5.3 语言主干的分层图与结构工作簿

状态：**首轮复核 1 项 P3 + 4 项 P2 + 2 项门禁测试迁移，以及导航跟进 P2 已修复，
等待 Codex 再次复核**。
本票只做 T05（GitHub #6）：按官方 `zai-org/GLM-5.3`
与本地官方 transformers `glm_moe_dsa` 实现，交付语言主干的整网→层组→Block→算子
分层 HTML、两页三组同源工作簿、真实语义检查与证据。未 commit/push、未关闭票、未进入
T06、未改自动化/协调状态；未运行/实例化模型、未下载权重。

## 接手确认

- 工作区/cwd：`/Users/xiechonghang/Workspace/LLMarchi_excel`。
- HEAD：`947ae78f120279403ec0a4c539e99d3d8869e1d6`（main，与派单一致）；本票未产生 commit。
- 接手快照：`/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T05-baseline-20261028-w8eh5d_w/manifest.json`
  （396 项，2026-10-28T17:29:22+08:00 采集；同目录 `files/` 为内容副本）。改动/新增均对照该快照。
- 快照后由协调流程写入的 `docs/handoffs/…`、任务目录 README/execution/monitor/tickets/
  T05 派单文件只读，未编辑；`changed-files.json` 将其列为 external coordination。

## 首轮复核修复（findings-round-1.md）

| 复核项 | 修复与证据 |
| --- | --- |
| **P3** 通用页面残留 DeepSeek 专属文案 | `PageProfile` 新增 `network_groups_label` 与 `legend`：DeepSeek 保持“四类层组”和自身真实机制图例；GLM 用“三组层型”和只含自身机制（Attention 内嵌 DSA Indexer、压缩 latent/indexer 键缓存）的图例；共享 `diagram_css` 的容器注释也改为模型中性。GLM 候选扫描 `四类/CSA/HCA/overlap/num_full_windows/N_win/Attention ⊃` 均为 0（verifier 断言），network.png 可见 GLM 图例；DeepSeek 图例保留其 CSA/闭窗说明。参数跳转路由未改。 |
| **P2** 稀疏掩码语义反向且 3D | `g_index_mask` 改为 **4D `[B,1,seqlen,Lidx]`**；语义按源码改为 `new_ones` 全 True → top-k 选中位置 scatter 成 **False**（True=屏蔽）；op 改为 `scatter+unsqueeze`；张量名/note、模块讲解、算子表、命名标签、检查与回归测试同源更新；`g_sparse_mask` 明确 `masked_fill(index_mask, min)` 与 bool `& ~index_mask` 等价。 |
| **P2** MoE 组掩码维度 | 拆成真实三步：`group_mask [Ttok,Ng]`（zeros+scatter）→ `expanded [Ttok,Ng,E÷Ng]`（unsqueeze+expand）→ **`score_mask [Ttok,E]`**（reshape）；`g_moe_group_masked_fill` 明确消费同型 `[Ttok,E]`；检查/测试断言三步顺序与最终维度。 |
| **P2** MoE token 路由依赖缺失 | 新增完整分配矩阵 `g_expert_mask [E+1,K,Ttok]`（`one_hot(top_k,E+1).permute(2,1,0)`）与 `g_expert_hit`（sum+nonzero）；`torch.where(expert_mask[expert_idx])` 显式输出 **`g_expert_topk_pos [N_e]` + `g_expert_token_idx [N_e]`**；select_tokens 用 token_idx gather；专家加权用 `top_k_weights[token_idx, top_k_pos]`；`zeros_like` 初值 + `index_add_(0, token_idx, …)` 回写。仍只画单份专家模板，`N_hit/N_e`/具体索引保持数据相关未知；检查与测试断言移除任一索引都会报错。 |
| **P2** 必要布局步骤不闭合 | KV latent RMSNorm 后新增显式 **3D→4D view**（`g_k_pass_view`，缓存 update 消费 `[B,1,seqlen,512]`）；AV 后新增 **`transpose(1,2).contiguous()`**（`g_attn_transpose` → `[B,seqlen,Nh,Dv]`），再 reshape 拼头；来源定位/图/表/讲解/检查同步。 |
| 2 项 `test_m2_guard` 门禁 | 迁移为公共入口 `run_hf_build(all_variants=True)`；仅替换可选 ONNX 依赖导入（`_load_onnx_stack`），真实逐模块 checker/shape/simplify 门禁保留：首个 SWA+Hash 组 `simplify`/`shape` 失败仍被具名拒绝；2 项通过，未跳过。 |

首轮修复候选：`d511bdfd…`（导航复核对象，保留）；导航修复后的最终候选：
`abefb101988c519dc81f5cf60aa27553e312cd0038d799701e4c460415456621`（见下）。
`9e34f204…`（初版）、`06c14b75…` 与首个正式候选 `8902247c…` 均保留，未修改。

## 导航跟进修复（findings-navigation-followup.md）

**P2：GLM 讲解/引用链接缺层组命名空间。** 根因是 `render/glm53_narrative.py:_C`
把链接作用域写死为 `"block"`，而页面在渲染前会把 Block id 作用域化为
`block-<组id>`；讲解里的 `#mod-/#p-/#t-` 因而缺组名，点击后退回 network。修复：

- `_C` 改为读取当前 Block 的真实 `block["id"]`（即 `block-<组id>`），不再硬编码；
  所有讲解/引用链接与参数表、张量表、模块详情使用同一已命名空间。
- 未改路由、未禁用链接、未复制页面；模块卡片/SVG 链接保持原路径。
- 新增检查：`test_glm53_structure` 与 `verify_glm53.py` 扫描全部
  `href="#..."`（普通锚点）要求目标存在、元素 id 唯一，并断言 3 组各自的
  `mod-block-<gid>-<module>`、`p-block-<gid>-Rk`、`t-block-<gid>-g_hidden_in`、
  `op-block-<gid>-g_input_norm` 均有真实 id 与链接；禁止未作用域的
  `#mod-block-glm_*`、`#t-block-g_*`、`#p-block-E`。
- 真实点击验证（headless Chrome 同源 iframe，DOM `click()`）：
  network → full-moe 层组 → Block → 点击“专家模板” →
  目标 `#mod-block-full-moe-glm_moe_expert` 打开并停在 ops 层；再点张量
  `g_expert_mask`、返回 Block、模块卡片、参数 `E`、返回整网；full-dense 的
  `glm_attn_q` 讲解同样进入 ops。结果：无错误、无缺目标、无重复 id，
  `navigation-click-check.json` 的 `ok=true`。工作簿字节未变（SHA `135105af…`），
  沿用 native 重算证据。

## 官方来源与配置身份（先锁定，再开发）

| 项 | 值 | 证据 |
| --- | --- | --- |
| 模型仓库 | `zai-org/GLM-5.3` | HF API `/api/models/zai-org/GLM-5.3` |
| 仓库 revision | `ba32f8c19c8738e2c57c71798c601d477a7bf844`（2026-10-14） | 同上；`rules/glm53_hf.py:MODEL_REPOSITORY` |
| 根配置 | `models/zai-org/GLM-5.3/config.json`，SHA256 `3ac72612…72e69` | 只取 config + 模型卡（README 14 KB），未取 141 个权重分片 |
| model_type / architectures | `glm_moe_dsa` / `["GlmMoeDsaForCausalLM"]` | 根配置；`load_hf_model_values` 双条件校验 |
| 实现 | 本地官方 transformers 检出 `hflib/transformers` HEAD `536ecc00…`（5.19.0.dev0），模块 `src/transformers/models/glm_moe_dsa/` | 模型卡指向该模块文档；模块 `@auto_docstring(checkpoint="zai-org/GLM-5")` 为同族基座 |
| checkpoint 保存版本 | config `transformers_version="5.15.0"` | 记录在 `data/effective_config.json`；本地 5.19.0.dev0 已含该模块 |
| 已分析输入 SHA | 7 项（config + modeling/configuration + cache_utils/masking_utils/modeling_rope_utils/activations） | `reports/source_manifest.json`，构建时逐项复算；漂移即具名拒绝 |

**身份辨伪**：`glm_moe_dsa` 是 GLM-5.3 官方仓库的 `model_type`（`pipeline_tag=text-generation`），
不是 GLM-5.3-Flash（`glm5_next`，`image-text-to-text`）也不是 GLM-5（同架构基座）。
本票没有把 `glm5_next`/DeepSeek 的事实套到 GLM-5.3 上；tests 对 DeepSeek 机制
（mHC/压缩器/Hash/tid2eid 等）有拒绝断言。

**实现版本与 config 记录版本的差异（明确，不混用）**：根配置记录
`transformers_version="5.15.0"`，而本地官方检出 HEAD 为 `536ecc00…`（5.19.0.dev0）。
两版 `glm_moe_dsa` 在实现上确有差异：5.15.0 的 MLA 缓存写展开后的 K/V、mask 字典键为
`deepseek_sparse_attention`；本地 5.19.0.dev0 写压缩 latent（512+64）、mask 键为
`indexed_attention`，并与本 config 的 `layer_types=indexed_attention` 一致。本票以
**本地官方检出的实际代码**为唯一实现版本（config 的 5.15.0 仅作为保存版本记录），
既没有把 5.15.0 的缓存结论混进本图，也没有声称两版等价；若复核要求改用 5.15.0，
需重新锁定该 tag 并按展开 K/V/不同 mask 键重做缓存与注意力语义。

关键静态解释（每条有源码行号证据，见 `rules/glm53_hf.py` 参数记录）：
- 78 层（0..77）；`indexer_types` 显式 78 项，与 `index_topk_freq=4`/
  `index_skip_topk_offset=3` 推导一致：full 层 21 个（0,1,2,6,10,…,74），shared 57 个。
- `qk_head_dim = 192+64 = 256`；`head_dim` 在 `__post_init__` 被改为
  `qk_rope_head_dim=64` —— 根配置保存的 `"head_dim":192` 不参与 RoPE（若按 192 解释会算错频率数）。
- MLA：KV 缓存存 **压缩 latent**（keys 512 + values 64），用时才对全部 Lkv 位置做
  `kv_b_proj` 展开；不是缓存展开后的 64 头 K/V。
- DSA indexer：full 层有 wq_b/wk/带 bias 的 LayerNorm/weights_proj 与
  `[B,L,128]` 键缓存；shared 层不实例化 indexer，复用上一层 full 的 top-k（层间传递）。
- MoE：sigmoid 打分 + `e_score_correction_bias` + 256 选 8 + 归一(+1e-20) + ×2.5；
  `n_group=1` 时组选择分支仍执行但单组掩码全通过（如实表达，不省略也不夸大）；
  共享专家宽度 2048，与 routed 并行相加。
- 残差：两处**普通相加**；没有 mHC/多流/Sinkhorn，也没有滑动窗口/压缩条目。
- MTP：checkpoint 的 `model.layers.78.*`（hnorm/enorm/eh_proj/shared_head）是训练期
  MTP 头；本地实现 `_keys_to_ignore_on_load_unexpected` 显式忽略且 forward 只循环 0..77，
  不在本主干（`data/network_overview.json` 的 `Nmtp` 参数明确标注）。
- 量化：根配置声明 FP8（e4m3, block 128×128）；本票只分析 HF 逻辑 eager 路径，
  不分析量化执行/kernel/数值。其 `modules_to_not_convert` 只列 full 层与 MTP 层的
  indexer/q_a/kv_a 等，可交叉印证 shared 层没有 indexer 参数；其中 `indexers_proj`
  是本地 HF 实现未定义的保存端条目（结构事实以已锁定 HF 源码为准，未读取权重索引）。

## 交付入口与真实产物

```bash
PY=/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
cd /Users/xiechonghang/Workspace/LLMarchi_excel
PYTHONPATH=structure_reader/src "$PY" structure_reader/scripts/read_model_structure.py \
  build-hf --model glm_5_3_hf --all-variants --json
```

- **最终候选**：`abefb101988c519dc81f5cf60aa27553e312cd0038d799701e4c460415456621`
- **真实 HTML**：[viewer/structure.html](../../../structure_reader/outputs/glm_5_3_hf/abefb101988c519dc81f5cf60aa27553e312cd0038d799701e4c460415456621/viewer/structure.html)
  （1,589,793 B；SHA256 `b9ab427d…ac41648`）
- **真实 Excel**：[tables/decoder_variants.xlsx](../../../structure_reader/outputs/glm_5_3_hf/abefb101988c519dc81f5cf60aa27553e312cd0038d799701e4c460415456621/tables/decoder_variants.xlsx)
  （179,044 B；SHA256 `135105af…b6500a`；仅注记顺序修正，工作簿字节未变）
- 同候选：`data/layer_structures.json`（505,488 B；SHA256 `7b6de5bf…25940a`，导航修复不改数据）、`data/network_overview.json`
  （51,809 B）、`data/effective_config.json`（66,358 B）、`reports/source_manifest.json`
  （7 项输入 + 26 项工具源码 + 身份）、`run_manifest.json`。
- **保留的历史候选**：首个正式候选 `8902247ce6a2e136136787f11b27603fa5a8c3c8f52630b77fb9d3e065cf4437/`；
  初版 `9e34f204…`（首轮复核对象，按要求保留）；首轮修复候选 `06c14b75…` 与
  `d511bdfd…`（导航复核对象）保留供对照。导航修复后的**最终候选**为 `abefb101…`。
- 重复执行同一命令：analysis_id 不变、`reused: true`，候选目录不增。
- 精确中间候选清理（仅本票路径，已记录）：`535d48e7…`、`212f79ff…`、`3acdf68f…`、`fc24c93f…`
  在最终候选发布后删除；接手前 DeepSeek 候选 `c3fd4296…` 与 T04 首轮候选 `dab11c89…`
  未动。

## 已实现

### 1. 分层数据（同一份参数化结构）

- `extract/glm53_hf.py`：三种真实层型组合（full+dense / shared+sparse / full+sparse）各一份
  Block 结构，shape 保留符号 `seqlen`/`kvcachelen`；每份 52/62/79 个算子、84/92/117 个张量。
  Attention（Q 低秩、交错部分 RoPE、压缩 latent 缓存、kv_b 展开、稀疏 mask、eager 核心、
  o_proj）、DSA indexer（full）/prev_topk（shared）、dense MLP 或 TopK MoE + 共享专家、
  两处普通残差均按所选源码逐条表达。
- `extract/glm53_network.py`：token→logits 整网（embedding、默认 position_ids、
  `create_causal_mask(..., allow_is_causal_skip=False)`、交错 RoPE、78 层真实顺序与三组归属、
  Final RMSNorm、logits 切片、lm_head）；层间 top-k 单向传递与 cache 读前/写后状态；
  MTP/视觉边界显式。
- 场景：`prefill-empty`、`prefill-symbolic`、`decode-nonzero(4096)`、`decode-long(131072)`、
  `decode-multi(4,4092)`；`Lkv=Lidx=Lmask=kvcachelen+seqlen`、`K_sel=min(2048,Lidx)`、
  `Ttok=B*seqlen`。不套用滑窗/压缩窗口，也不把 Decode 固定为 +1。

### 2. 分层浏览页（复用同一渲染核心）

- `viewer/structure.html`：10 个层级 section（整网 + 3 层组 + 3 Block + 3 算子层），
  45 个 SVG、44 段模块讲解、293 个张量行；默认只显示当前层级，面包屑/链接可逐级进入与返回；
  场景按钮切换 5 个预设；源码快照、参数/张量/图/算子联查沿用 T03/T04 的锚点体系。
- 模型 profile（`render/hf_model_profile.py`）把层型短名、层组说明、indexer 模式表、
  模块分组、总览布局、模块讲解与源码命名表从通用页面装配中分开；DeepSeek 默认 profile
  保持既有输出（下方逐字节证据）。
- 三种组分别显示真实层索引与次数（3/57/18，共 78），模块清单与图随 full/shared indexer
  与 dense/MoE 真实差异变化（shared 组无 indexer 模块/缓存）。

### 3. 同源两页工作簿

- `tables/decoder_variants.xlsx`：`Prefill`/`Decode` 两页，每页 3 组、193 个算子行
  （52/62/79），来自与 HTML 同一份 `hf_layer_structure.v1`。
- 每页顶部可填写性能配置（seqlen 8/1、kvcachelen 0/4096、batch 1、精度、位宽、
  专家命中数未知）；派生参数 `Lkv/Lidx/Lmask/Ksel/Ttok` 是引用本页输入/本组参数单元格的
  活公式（带 `IF`/`ISNUMBER`/`IFERROR` 守卫），并新增实际输入/输出/权重 shape 公式列。
- 数据相关量 `N_hit`/`N_e` 写“未知（原因）”，不写 null/0；成本列统一“待提供”，
  无猜测耗时、无成本 SUM；层组循环次数来自真实层索引，不复制 78 层或 256 专家。
- 整网页尾列出网络外围 8 个算子、三组真实层索引/次数（3/57/18，共 78）、MTP 与视觉边界。

## 复用 / 重构 / 删除 / 新增

### 保留复用

- `build_hf_decoder.py` 的资料核验/身份/内容寻址发布/失败隔离；`cli._materialize`。
- `extract/hf_scenario.py` 的场景投影与三态求值；`scenario_evaluator.py` 白名单与 Excel 编译。
- `render/diagram.py` 的真实张量引用建图/SVG/锚点；`render/detail_render.py` 的算子卡/
  张量/参数联查；`render/hf_structure_page.py` 的层级/路由/场景条。
- `export/operator_workbook.py` 唯一 Workbook 创建/sheet/保存/稳定 ZIP；T04 的活公式、
  实际 shape、未知/恢复与显式引用消歧。

### 原位重构（迁移调用后删重复）

| 文件 | 改动 |
| --- | --- |
| `build_hf_decoder.py` | 引入 `_Adapter` 注册（DeepSeek/GLM 各一）：共同编排、来源/工具 SHA、检查、渲染、导出只一处；不再把 DeepSeek 的 `SUPPORTED_MODEL_KEYS`/README/ONNX 组数写死在流水线。单层路径保持 DeepSeek；GLM 单层/ONNX 显式拒绝。 |
| `extract/hf_layer_structures.py` | 容器构造参数化（loader/groups/build_documents），DeepSeek 默认不变；可选携带 `indexer_type`。 |
| `extract/hf_scenario.py` | 可选输出 `model_revision`/`indexer_type`（仅模型声明时写键），DeepSeek JSON 字节不变。 |
| `checks/deepseek_v4_hf_contracts.py` | 通用骨架/数据流/输入身份检查提炼到新 `checks/hf_contract_core.py`；DeepSeek 语义锚点保留。 |
| `export/hf_workbook.py` | 模型无关的参数行/输入/组声明/整网归属提炼到 `export/hf_workbook_common.py`；本文件退为 DeepSeek 薄适配。 |
| `render/hf_structure_page.py` | 增加 `profile` 参数；模型文案/分组/总览/讲解/层格/层型表由 profile 提供；源码命名表用 `use_source_table` 临时启用、退出复原。 |
| `render/diagram.py` | `build_overview_graph(positions, frames)` 支持模型自定义布局；`modeling_file()` 替代硬编码 DeepSeek 文件；默认路径字节不变。 |
| `render/diagram_source_labels.py` | 新增可切换的源码命名表上下文（默认仍为 DeepSeek 表）；补 LayerNorm/MaskedFill 等真实类型名。 |
| `render/display_names.py` | 新增 GLM 参数 id→配置字段显示名（Dq/Dn/Dv/Rkv/Is/Ng/Ktg/EpsA/Lidx/Ksel/Ttok）。 |
| `cli.py` | `--model` 帮助列出两个模型键；`--layer-index` 缺省按模型适配；输出按实际组数/层号。 |
| `structure_reader/README.md`、`docs/hf_reading_variants_contract_v1.md` | 增加 GLM-5.3 入口、身份/边界与 T05 增量说明。 |

### 新增

| 文件 | 职责 |
| --- | --- |
| `rules/glm53_hf.py` | 官方身份/revision、7 项源码指纹、配置静态解释、3 组归属、参数来源记录、GLM 场景。 |
| `extract/glm53_hf.py` | 三种层型组合的参数化 Block 结构（MLA/DSA/MoE/残差/缓存）。 |
| `extract/glm53_network.py` | 78 层 token→logits 整网总览与层序/组归属。 |
| `checks/glm53_hf_contracts.py` | GLM 语义检查：身份/参数事实/Attention/缓存/DSA/MoE/残差/场景/输入身份，并拒绝 DeepSeek 机制。 |
| `checks/hf_contract_core.py` | 两模型共用的骨架/数据流/输入身份检查。 |
| `export/hf_workbook_common.py` | 工作簿公共声明（输入行、参数行、组、整网归属、两页框架）。 |
| `export/glm53_workbook.py` | GLM 薄适配：两页三组、派生映射、真实层归属校验、成本待提供。 |
| `render/hf_model_profile.py` | DeepSeek/GLM 页面 profile（默认 DeepSeek 保持自身文案与图例；层组数文案/图例/分组/布局由各模型声明）。 |
| `render/glm53_narrative.py` | GLM 模块/总览中文讲解（按 full/shared、dense/MoE 分支）。 |
| `render/glm53_source_labels.py` | GLM 算子源码表达式/张量短名/行号；页面临时启用。 |
| `tests/test_glm53_structure.py` | 30 项回归：身份/层组/Attention/缓存/DSA/MoE/残差/稀疏掩码/场景/网络/契约拒绝/产物/工作簿/profile/锚点闭合。 |
| `T05-verify/verify_glm53.py`、`workbook-check.json` | 真实候选 + LibreOffice 重算 + 编辑链 + 锚点闭合/唯一性独立验证。 |
| `T05-verify/click_navigation_check.py`、`navigation-click-check.json` | headless Chrome 真实 DOM 点击：Block/讲解模块/参数/张量/返回/另一组。 |
| `T05-verify/changed-files.json`、`incremental.patch` | 相对接手快照的改动/新增证据（含外部协调文件与接手前改动分类）。 |
| `T05-verify/network.png`、`block.png`、`ops.png`、`module-glm_attn_mask.*`、`module-glm_attn_cache.*`、`module-glm_moe_route.*`、`module-glm_moe_expert.*` | 真实候选/模块图（含修复后的掩码/路由/缓存布局）的 headless Chrome 渲染证据。 |

### 删除 / 退役

- 本票未删除产品文件；仅删除本票精确中间候选目录（见上）。
- DeepSeek 单层入口、两页四组工作簿、T03 分层页与可选 ONNX 路径保留；未退役旧 reference/DSpark。

## 实际验证证据

### 相关测试（本票环境）

| 测试 | 数量 | 结果 |
| --- | --- | --- |
| `tests/test_glm53_structure.py`（本票新增，含首轮/导航修复回归） | 30 | OK |
| `tests/test_hf_decoder_reading.py` | 61 | OK |
| `tests/test_hf_layer_levels.py` | 41 | OK |
| `tests/test_hf_network_overview.py` | 21 | OK |
| `tests/test_operator_workbook_v2.py` | 24 | OK |
| `tests/hf/test_operator_workbook.py`（v1 兼容） | 57 | OK |
| `tests/test_hf_build_missing_xlsx.py` | 2 | OK |
| `tests/test_m2_toc.py` | 5 | OK |
| `tests/test_scenario_evaluator.py` | 45 | OK |
| `tests/test_page_topology_r2b.py` / `r2a` / `r1` | 30 / 27 / 17 | OK |
| `tests/test_page_js_execution.py` | 2 | OK |
| `tests/test_model_registry.py` | 15 | OK |
| `tests/test_m2_guard.py`（迁移到公共入口） | 2 | OK |

合计 **349 项既有直接相关 + 30 项 T05 新增 = 379 项通过**。首轮复核指出的
2 项 `test_m2_guard` 已迁移到公共 `run_hf_build(all_variants=True)` 并通过（见下）。另 `tests/test_hf_decoder_diagram.py`：
43 项中 37 通过、6 项失败；已用接手快照 `files/` 原样源码复跑，**同一 6 项在接手时已失败**
（短符号 `N_win`/`（语义id）`/`B,T,Rq` 等旧布局断言早于本票被完整名称替换），不是本票回退；
按“既存旧布局漂移如实列出”保留，不修改该历史测试。`test_hf_decoder_onnx.py`/`test_m2_onnx.py`
在本环境缺 onnx，未计为通过；本票未改 DeepSeek ONNX 路径。

### DeepSeek 不回退（逐字节对照 T04 最终候选）

用本票代码重新构建 DeepSeek `--all-variants`（内存字节，不发布新候选），与
`c3fd4296…` 逐字节比较：

| 产物 | 结果 |
| --- | --- |
| `data/layer_structures.json` | 完全相同（793,560 B） |
| `data/network_overview.json` | 完全相同（53,152 B） |
| `tables/decoder_variants.xlsx` | 完全相同（262,368 B，SHA `2da0caf1…`） |
| `viewer/structure.html` | 除候选 id 与共享 CSS 的中性注释外相同（2,694,048 B） |
| `README.md` | 除候选 id 外相同 |

说明：本票提炼共享模块后，DeepSeek 的 **工具源码身份/analysis_id 会变化**（源码已改），
但结构数据与工作簿字节一致；页面仅共享 CSS 的中性注释与候选 id 不同（可见内容无变化，
DeepSeek 图例仍是其自己的 CSA/闭窗说明）；原 T04 候选目录未重写、未删除。

### GLM-5.3 真实候选独立验证

[`workbook-check.json`](T05-verify/workbook-check.json)：`status=passed`，对象是落盘候选
（不是内存 spec）：
- `run_manifest` 8 个交付 SHA、7 项输入 SHA、26 项工具 SHA 全部复算一致；identity 规范化
  重算 SHA256 == analysis_id。
- 数据：三组 3/57/18 恰好覆盖 78 层；full indexer 21 层；dense 0/1/2；Block 算子
  52/62/79、张量 84/92/117；各参数与
  `config.json` 独立核对（含 Dq=256、Dk rope=64、V=256、E=256、K=8、Es=2048、Is=12288、
  Scl=2.5、theta=8e6、两套 eps）；Attention/缓存/DSA/MoE/残差语义与“无 DeepSeek 机制”全部通过。
- 场景独立期望：Prefill(8,0) `Lkv=8,Ksel=8`；Decode(1,4096) `Lkv=4097,Ksel=2048`；
  长历史(131072) `Ksel=2048`；多 token(4,4092) `Lkv=4096`；符号场景 unknown。
- HTML：10 section、45 SVG、44 讲解、293 个张量行、78 层真实顺序格、全部层级链接/场景 id/源码快照；
  1,346 个 id 无重复、3,151 个普通锚点全部有目标（verifier `navigation`）；
  无 `swa-hash`/`csa-topk`/`DeepseekV4`/`mHC 权重`，也无 `四类/CSA/HCA/overlap/num_full_windows/N_win/Attention ⊃`；
  页面显示“三组层型”与 GLM 自己的图例。
- 工作簿结构：2 页 × 3 组 × **193 行**（52/62/79）；输入/派生公式/实际 shape/
  循环归属/成本待提供/数据相关未知；无成本 SUM；无 DeepSeek 组 id。
- **LibreOffice native 重算**（同一现成文件，不重新导出）：
  - 原场景：Prefill `8/8/8/8/8`，Decode `4097/4097/4097/2048/1`；
  - 实际 shape：`g_input_norm` 输入 `[1,8,6144]`、`g_kv_b_proj` 输入
    `[1,1,8,512]`、indexer 键缓存输出 `[1,8,128]`；Decode 分别 `[1,1,6144]`、
    `[1,1,4097,512]`、`[1,4097,128]`；
  - 清空 seqlen → `Lkv/Ksel/Ttok` 为“未知（依赖参数…）”，不是 0；填 `abc` → 未知且无
    `#VALUE`；同一文件改回 8 → 恢复 `8/8/8`；填 -1 → 未知；
  - 改 `seqlen=4, kvcachelen=4092, B=2` → `Lkv=4096, Ksel=2048, Ttok=8`，实际 shape
    `[2,4,6144]`、`[2,1,4096,512]`；全表扫描无 Excel 错误值。
- 重算后端为 LibreOffice native（提示缺 Calibri 字体，不影响数值）；**未做 Microsoft
  Excel 原生 GUI 打开验证**。

### 页面渲染证据（headless Chrome，非用户验收）

- [整网页](T05-verify/network.png)、[Block（full-moe）](T05-verify/block.png)、
  [内部算子索引](T05-verify/ops.png)；
- [稀疏掩码模块图](T05-verify/module-glm_attn_mask.png)（True=屏蔽、4D）、
  [Attention 缓存模块图](T05-verify/module-glm_attn_cache.png)（norm→4D view→cache）、
  [MoE 路由模块图](T05-verify/module-glm_moe_route.png)（组掩码 [Ttok,E]）、
  [MoE 专家模块图](T05-verify/module-glm_moe_expert.png)（token_idx/top_k_pos 贯通）
  （同名 `.html` 可直接打开）。
- 已用 URL hash 实际加载 `#/group/<gid>`、`#/group/<gid>/block`、`#/group/<gid>/ops`
  并截图；另用 headless Chrome 同源 iframe 做**真实 DOM click**：整网→full-moe 层组→
  Block→“专家模板”讲解→张量→返回 Block→模块卡片→参数 E→返回整网，再在 full-dense
  点击 `glm_attn_q` 讲解；11 步均记录 hash/可见层级，`ok=true`、无缺目标/重复 id
  （[navigation-click-check.json](T05-verify/navigation-click-check.json)）。
  未做人工鼠标/键盘/窄屏阅读。

## 未验证 / 明确限制

- 用户性能公式与 XPU 能力仍未提供：无算子耗时/组小计/整网串行汇总；成本列“待提供”，
  不猜、不记 0、不做父子重复计数。
- GLM-5.3 **未实现可选 ONNX**：显式 `--onnx` 具名拒绝（不虚构、不静默忽略）；
  DeepSeek ONNX 路径未改、未回退。
- GLM-5.3 **只提供分层入口**（`--all-variants`）；单层长页未实现，缺该参数时具名拒绝。
- 只覆盖语言主干：视觉/图像输入路径不在本票；MTP 层 78 不执行；FP8 量化执行、
  flash_mla indices 核路径、2D padding batch packing、generate 采样循环、loss/aux_loss
  未分析；不做数值等价。
- 未在原生 Microsoft Excel 中做 GUI 阅读验证；未做真实设备性能/Profiler/并行融合通信。
- 静态工程与 headless 渲染通过不代表用户阅读认可；需 Codex/用户复核。

## 保护与停止

- 未 reset/clean、未批量格式化、未恢复根两项删除、未改接手前未提交改动；快照中被本票
  改动的 18 个文件与新增 11 个产品文件见 [changed-files.json](T05-verify/changed-files.json)
  与 [incremental.patch](T05-verify/incremental.patch)。
- 接手前 DeepSeek 候选 `c3fd4296…`、T04 首轮候选 `dab11c89…`、T03 候选及证据未动；
  本票首个正式候选 `8902247c…`、初版 `9e34f204…`、首轮修复候选 `06c14b75…` 与
  导航复核对象 `d511bdfd…` 保留；
  仅删除本票更早的中间候选 `535d48e7…`/`212f79ff…`/`3acdf68f…`/`fc24c93f…`。
- 协调/自动化文件只读，未编辑；定时检查保持 PAUSED。
- **完成即停止，等待 Codex 复核**：不 commit/push、不关闭 #6、不进入 T06。
