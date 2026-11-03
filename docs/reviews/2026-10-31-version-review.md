# 当前版本成果评审（T01–T14，2026-10-31）

## 范围与方法

- 固定点：`947ae78`（2026-10-22）。此后**没有 commit**，全部成果在工作区；为让 untracked 文件进入 diff，先 `git add -A` 暂存（评审后已 `git reset` 还原）。
- 评审 diff：`git diff --cached 947ae78 -- structure_reader/src structure_reader/tests structure_reader/docs structure_reader/tools structure_reader/README.md`，118 文件、+56,661/−1,015 行。
- 不含：`structure_reader/outputs/`（1.2 G 交付产物，gitignore）；本轮不重跑 browser-check（用户已跑 81/81）。产物侧用现有 manifest/relations/xlsx 哈希抽查。
- 方法：`/code-review` 两轴并行子代理；下面对每条关键结论做了人工复验，复验结论标在“复核”中。

---

## Standards

### 硬违规（有书面标准）

1. **绝对主机路径进入身份与产物。**
   `tool_identity()`（`render/archify_preview.py:186-204`）把 `"cli": str(tool.cli)`、`package.root` 绝对路径和本机 `node --version` 放进 identity；`archify_identity` 进入 `analysis_id`（`build_hf_decoder.py:1593-1596`）并写入 `data/archify/manifest.json`（`archify_layers.py:2651`）。
   书面标准：`developer_guide.md:42` 禁止“绝对工作区路径…解释器安装路径等偶然字段”进入 analysis_id；`output_reference.md:46` 要求“产物不含绝对路径、时间戳、解释器安装路径”。
   **复核：属实。** 四个新候选与 T13 DeepSeek 的 `data/archify/manifest.json` 均含
   `/Users/xiechonghang/.dsh/skills/archify/bin/archify.mjs`；HTML 不含该路径。
   影响：同一内容换机器/换技能安装路径会得到不同 `analysis_id`，违背“同输入字节级确定”。

2. **Archify 产品源码清单漏掉 `semantic/axes.py`、`shape_expr.py`、`layout_probe.py`。**
   `build_hf_decoder.py:1393-1408` 的基础清单 + DeepSeek 扩展只列 `semantic/rules.py`、`bundle.py`、`extract/deepseek_v4_semantic.py`；而 `semantic/rules.py:51-52` 导入前三个模块，DeepSeek normalized bundle 在 build 时走这条链（`build_hf_decoder.py:1292`）。
   **复核：部分属实，需修正范围。** 四个 T14 新模型不导入 `semantic/*`（`archify_layers.py` 无该 import），所以影响面是 **DeepSeek/T13 重建路径**，不是四个新候选。改动 `axes.py` 可能改变 DeepSeek normalized 图而 `analysis_id` 不变，正是 `developer_guide.md:41` 说的伪复用。

### 判断项（无书面标准，属坏味道）

- **Divergent Change / 巨型模块**：`archify_layers.py` 新增 4,233 行，同时承担图投影、布局、文档 schema、身份/manifest、临时目录+finalize、HTML wrapper、审计，以及第二套 profile 驱动构建器（3329-4233）。10 个函数超过 140 行，最长 `build_moe_graph` 305 行。
- **重复代码**：`_sha256_file` 5 份（cli/inventory/build_hf_decoder/archify_layers/archify_preview）；`_component_id` 逐字重复（2015/835）；`_sides` 几何重复（1993/963）；`ViewNode` 与 `LevelNode` 是同一 7 字段图模型。
- **Data Clumps**：五个 `_Adapter(...)` 各约 25 个关键字实参；`hf_model_profile.py` 五处 38 个 per-model hook；`archify_profiles.py` 四段共享同一 segment 骨架。
- **Speculative Generality**：`PreviewRequest`/`generate_previews`（约 130 行）零调用；`_boundary_tensor_ids`、`_first_op_using`、`_producer_of`、`_key_tensor` 无引用；`build_hf_decoder.py:65/67` 的 `ARCHIFY_REPOSITORY`/`ARCHIFY_LEVELS` 定义后从未使用（本条为本次复验补充）。
- **状态/缓存形状重复**：`_EXPR_LKV/_EXPR_LIDX/_EXPR_VQ` 在 Flash/Qwen 各一份；五个 narrative `_C` 类。

**符合项**：调度是 registry/profile 驱动（仅 3 处 `model_key ==`）；页面 JS 只渲染预计算派生；错误保持具名、traceback 只进失败记录。

---

## Spec（对照 T14 `14.md` 与 vNext 需求）

### (a) 缺失/部分实现

1. **索引与候选 README 的构建命令不能复现候选。**
   `structure_reader/docs/model_delivery_index.md:95` 和每个候选 README 都给
   `build-hf --model <key> --all-variants --json`；但 `run_hf_build` 的 `archify_layers=False`（`build_hf_decoder.py:1141`），`--archify-layers` 是显式开关（`cli.py:1181`），索引里 0 次出现。
   照文档重跑会得到**不含图、analysis_id 不同**的另一份候选。
   **复核：属实。** 违背“交付索引闭合/可复现”。

2. **没有图 ↔ Excel 节点粒度登记。**
   `14.md` 要求“图与旧表节点粒度的对应/差异明确登记”。索引只断言“图的新粒度不改变 Excel 行”；`T14-decoder-verify/round-2/node-map.json` 是“图节点→源码行”，不是→worksheet 行/计费项。
   **复核：属实。** Excel 5/5 当前 xlsx 哈希与 T11 相同，字节保真成立，缺的是映射登记。

3. **四模型 normalized 只是“全部拒绝”。**
   `normalization_audit.record()` 硬编码 `status:"rejected"`（`archify_layers.py:3269`）；非 DeepSeek 没有调用任何变换。11/11 Attention 的 raw/normalized 组件与连线完全相同。
   **复核：属实但不算违规**——规范允许 0 条应用（“无法证明就保留”）；这是能力缺口/审计形态问题，可列为改进项。

4. **主图卡片仍用缩写。**
   `_level_cards`（`archify_layers.py:2256`）的“符号与参数（全称）”卡片仍以 `B`、`H`、`Nh`、`Dq`、`Rk`、`Rq` 起头；Qwen 缓存卡片写 `[B,Nkv,Lkv,Dh]`。规范 §3：“主图不再用 B/S/H/Dh/Rk/Rq 作为这些参数的展示缩写”。
   **复核：属实。** round-2 的别名扫描只覆盖节点/边，没覆盖 cards；是否算“主图”存在口径争议（卡片确实同时给出全称）。

### (b) 范围蔓延

未发现 T14 专属的范围蔓延；大块新增属已验收的 T12/T13 模块与 T05–T11 适配器，ONNX/M2 变更属更早票。

### (c) 实现了但不对

5. **lane 路由的箭头丢维度/张量标签。**
   `archify_layers.py:2082-2088` 在 `via` 分支直接 `continue`，跳过了后面的 `edge.shape`/`edge.label` 写入。五个当前候选共 **830 条 data connection** 有 `via` 且无 label（DeepSeek 174、GLM 137、Flash 228、Qwen 244、HiDream 47）。
   例：GLM `attention-full-dense-raw` 的 `o-in→o-g_qk`/`o-in→o-g_av`，relations 里有 `g_key`/`g_v` 和全称 shape，渲染箭头两者都没有。
   **复核：属实且正是目检时看到的长线无标签。** 规范 §3 要求箭头默认标出形状；`test_archify_layers_models.py:142` 只检查 `graph.edges`，没检查 `candidate["connections"]`。

6. **HiDream ViT 组循环数标成 36。**
   `build_group_graph_for`（`archify_layers.py:3856`）对所有组都用 `network.scope.layer_count`；`group-vit-tower.json` 的 `n-g-loop` 写“36 层循环”，同图卡片又写“本组 27 层”，自相矛盾。
   **复核：属实。** 规范要求“各模型真实层组、循环次数保持”。

**无法从代码验证**：browser-check（用户侧 81/81 证据，本轮未重跑）。**已复核**：Excel 5/5 字节保真。

---

## 一页汇总

| 轴 | 结论 |
| --- | --- |
| Standards | 2 项硬违规（绝对路径进身份、DeepSeek 语义源码清单缺口）+ 5 类判断项；最严重：绝对路径使候选身份不可跨机复现 |
| Spec | 4 项缺失/部分（文档不可复现、无 Excel 粒度登记、normalized 仅拒绝、卡片缩写）+ 2 项实现错误（830 条箭头丢标签、ViT 循环数 36 vs 27）；最严重：830 条箭头违反“箭头默认标形状” |

这两项“实现错误”是**已验收产物里的真实缺陷**，现有人工目检与自动检查都没覆盖到；按你之前的说法，正好属于“后续整体复查”的范围。

---

## 后续建议（不属于 code-review 两轴）

### 优先修（建议 T15 或重开 #15）

| # | 问题 | 修法 | 代价 |
| --- | --- | --- | --- |
| 1 | 830 条 lane 边无标签 | `via` 分支补 `label`/`shape`（可在路由中点放标签）；测试改为断言 `candidate["connections"]` | 低；重建 5 候选 |
| 2 | ViT 组循环数 | 用该组真实层数，不用 `network.scope.layer_count` | 低 |
| 3 | 绝对路径进身份 | identity 用 `{discovery, version, cli_sha256, package.digest, node}`，去掉 `cli`/`package.root` 绝对路径；回归测试“换 HOME 路径 identity 不变” | 中；analysis_id 全变，需重建+重跑 81 图 |
| 4 | 索引不可复现 | 索引/README 命令补 `--archify-layers`；加“重建复现”测试 | 低 |
| 5 | 无 Excel 粒度登记 | 生成 graph-node → worksheet row/计费项的映射表并入库 | 中 |
| 6 | 卡片缩写 / normalized 仅拒绝 | 卡片改全称（或删符号图例）；normalized 要么实现至少一条可证规则，要么在索引明确“四模型无化简能力” | 低/中，需你定口径 |

### 工程卫生

- `archify_layers.py` 拆分（投影/布局/身份/编排/wrapper），先做“提取模块 + 保测试绿”；
- 合并 `_sha256_file`、`_component_id`、`_sides`、`ViewNode/LevelNode`；
- 删除 `generate_previews`、四个死私有函数、`ARCHIFY_REPOSITORY/ARCHIFY_LEVELS`；
- 仓库目前**无 lint、无 CI、无 pre-commit**（`pyproject.toml` 只有打包配置），这也是这些死代码/重复代码能留下来的原因：建议先加 `ruff + pytest` 最小 CI。

### 流程

- 修之前先做一次本地 commit 作为 fixed point（否则下一轮 review 仍然没有 diff 基线），产物另做归档（1.2 G，git 不收）。
- 修完按 T14 同样标准重跑：候选重建 → 81 图 browser-check → 证据/索引更新 → 人工目检。
