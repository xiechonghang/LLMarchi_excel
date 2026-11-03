# T10 结果：DeepSeek 完整 Excel 的 CUBE/VEC 粗估与串行汇总

状态：**实现与验证完成，停止等待 Codex 复核**。未关闭 GitHub #11，未进入 T11（GLM/GLM-Flash/Qwen/HiDream 性能），未 commit/push，未改协调文件与自动化（保持 PAUSED/45 分钟）。

## 接手核对

- 工作区 `/Users/xiechonghang/Workspace/LLMarchi_excel`；`main` HEAD `947ae78f120279403ec0a4c539e99d3d8869e1d6`（本票未产生提交，HEAD 未变）。
- 快照 `/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T10-cube-vec-baseline-20261029-nclgpfg5/manifest.json`：682 文件，逐文件 SHA256 比对，**682/682 存在且一致，唯一差异是 `docs/tasks/model-structure-2026-10-28/execution.json`**（快照后 Codex 协调状态写入，本票只读）。
- 本票增量（[baseline-diff.json](T10-cube-vec-verify/baseline-diff.json)）：修改 12 个（4 个协调文件只读、1 个交付索引、5 个产品源码、2 个测试）；新增 21 个中有 17 个由本票写入（1 个产品规则模块、12 个证据脚本/结果/渲染图、本报告等），另 4 个是 Codex 复核目录 `T10-cube-vec-review/`（本票未写）；删除 0。根 `CONTEXT.md` 与 `2026_10_17_workspace_structure_review.md` 的既有删除仍保持；旧候选、五模型产物与旧单算子模板零改动。

## 交付物

- 当前候选（**最终**，绑定冻结代码）：`structure_reader/outputs/deepseek_v4_flash_0731_hf/dcbf7eae3a26a0f7745c76b884bf55ae8406f7fe6d5631151075516ef09f8280/`
- 工作簿：[`tables/decoder_variants.xlsx`](../../../structure_reader/outputs/deepseek_v4_flash_0731_hf/dcbf7eae3a26a0f7745c76b884bf55ae8406f7fe6d5631151075516ef09f8280/tables/decoder_variants.xlsx)，SHA256 `d526126e75cf65dcd3f3b98da43687bc16fced4337a146d232ddd00a7d38d8ad`，401,028 字节，`Prefill`/`Decode` 两页、27 列。
- 分层浏览入口：[`viewer/structure.html`](../../../structure_reader/outputs/deepseek_v4_flash_0731_hf/dcbf7eae3a26a0f7745c76b884bf55ae8406f7fe6d5631151075516ef09f8280/viewer/structure.html)（结构页语义未变，SHA `9f4f7df0…e04645`）。
- 过程候选 `61b74593…`（T09 结构候选）、`b5cfa51d…`（Codex 首轮审阅的早期性能版）、`9d388880…`（Codex 第二轮审阅的版本）全部保留不覆盖；三者都不是当前候选。Codex 的 `first-native-observations.json` 对应 b5cf、`native-second-observations.json` 对应 9d，均不冒充本最终版证据。
- 证据目录：[`T10-cube-vec-verify/`](T10-cube-vec-verify/)（规则清单、原生重算改表、跨模型回归、候选闭包、测试汇总、渲染图）。

## 用户规则与接入位置

用户 2026-10-29 确认：含矩阵计算的算子按 CUBE，其余按 VEC；**连续按比例、不向 64 取整**。

- `T_cube_ms = B·M·K·N / 吞吐`（基准 B1/M1/K64/N64 = 1 ms ⇒ 默认 4096 MAC/ms，不是 FLOP/ms，不再乘 2）。
- `T_vec_ms = M·N / 吞吐`（基准 M1/N64 = 1 ms ⇒ 默认 64 元素/ms）。
- 两个吞吐是每页性能配置区里**可改的数值单元格**；成本公式逐级引用：耗时 → 工作量 → 本行维度单元格 → 本组参数区/本表输入。改 B/seqlen/kvcachelen/吞吐/`N_hit` 后由 Excel 重算。
- 唯一产品链：`export/operator_workbook.py`（通用写表器）+ `export/hf_workbook_common.py`（HF 公共声明）+ `export/hf_workbook.py`（DeepSeek 适配）。新增 `export/deepseek_perf_rules.py` 只**声明**分类/维度/约束/来源，不解析模型、不写 xlsx；已进入 `_DEEPSEEK_TOOL_SOURCES`，`run_manifest` 工具清单含其 SHA。
- 性能区是**声明式可选**能力：未声明 `performance.engine` 的工作簿（其余四模型）输出字节完全不变；DeepSeek 两页与单层兼容表都启用同一套规则。

## 列与汇总结构（不新增重复页）

每页顺序：标题/口径 → 性能配置与场景输入（含 `Tcube`/`Tvec`、`N_hit`）→ 四类层组（参数区 + 算子区）→ 整网外围块 → 整网串行汇总。

算子区在原有 19 列结构信息（算子 id/模块/输入输出权重 shape/参数联查/源码/循环归属/实际 shape）后追加 8 列：分类、CUBE B、M、CUBE K、N、工作量（MAC 或元素）、成本状态、性能假设与维度映射；既有的“成本”列改为用户规则耗时 ms。层组内 `mask_model_build`/`rope_main`/`rope_compress` 行显示“整网一次性项：见 net_*（不计入本组小计）”，成本单元格不写数值。

层组末尾：`单层算子小计（N 个计费行）` = `IF(COUNT(计费区间)=N, SUM(...), "未知（k 项未定）")`（只覆盖计费行，排除整网一次性引用行与容器行）；`层组总计 = 单层小计 × 真实层数`（只作阅读核对）。整网串行汇总表：入口外围 6 项 → `swa-hash`×2 → `csa-hash`×1 → `hca-topk`×20 → `csa-topk`×20 → 出口外围 7 项；列含义固定为 **F=单次成本（层组=单层小计、外围段=本段小计）、G=次数（真实层数或 1）、H=F×G**，整网只乘一次；汇总区给“已知部分小计（仅已计项；不是整网总计）”“未定贡献项数”“整网单次调用总耗时 ms（全部已定才是数值，否则给缺口数）”。

## 分类与维度规则（可审查清单）

完整清单见 [rules-coverage.json](T10-cube-vec-verify/rules-coverage.json)（逐算子类别、维度表达式、项数/重复、条件、计费范围、来源说明、符号是否在本组声明）。覆盖 **291 个算子实例**：`swa-hash` 58（CUBE 16 / VEC 42）、`csa-hash` 76（20/56）、`hca-topk` 68（18/50）、`csa-topk` 75（20/55）、整网外围 14（CUBE 2 / VEC 11 / 容器 1）；**缺少规则的算子 0，规则引用未声明符号 0**。

- Linear（权重存 `[N,K]`、`Y=X·Wᵀ`）：`B=batch, M=序列/行数, K=in, N=out`；batch 只出现一次（`q_a_proj`/`q_b_proj`/`kv_proj`/`o_b_proj` 等）。
- 注意力：`qk_scaled`/`av_matmul` 用 `B=B·Nh`（头数作矩阵 batch），`M=seqlen`，`K`/`N` 为 `Dh` 与可见 KV 轴；SWA 的 KV 轴是 `Lkv`，CSA/HCA 是 `Lkv+Ncomp`（cache.update 之后拼接压缩条目）。`repeat_kv` 按 `[B,Nh,KV,Dh]` 物化规模计 VEC。
- 分组线性 `o_a_proj`：`torch.bmm` 的 `B=B·G`、`K=Nh·Dh/G`、`N=Ro`，不把 G 再乘 M；`o_b_proj` 是 `G·Ro→H`。
- mHC：`attn_hc_project`/`ffn_hc_project` 是 `Nhc·H→(2+Nhc)·Nhc` 的一次线性；`*_hc_comb` 按其逻辑输入 `[B,seqlen,Nhc,Nhc]` 元素数计一次（Sinkhorn 的内部行列归一不另收费，与 Norm/Softmax 同口径，已在假设列声明）；`*_residual_mix` 是 `B=B·seqlen, M=K=Nhc, N=H` 的批量矩阵乘。
- 复合算子：矩阵部分计入 CUBE 并**列出所有矩阵项**——`shared_gate_up`（gate/up 两个同形 Linear，repeat=2）、`c_kv_gate_proj`/`i_kv_gate_proj`（两个同形 Linear，repeat=2）、`i_scorer`（q·kᴵᶜᵒᵐᵖ 一项 + weights_proj 一项，分两行显示、工作量求和）。伴随的 view/transpose/add/激活/clamp 明确注明不另收费。
- 卷积/scan：本模型 DeepSeek 主干不含卷积或以名字猜测的 scan；压缩器/Indexer 的“汇聚、RoPE、窗口拼接、block_bias”都按源码实际工作张量计，未用 `matmul` 字符串命中。
- MoE：不填运行期每专家 token 分布；专家 Linear/clamp/act/加权/汇合按**源码可证明的总路由分配数** `B·seqlen·K`（每 token K 个专家分配，TopK 与 Hash 同构）表达总量。`moe_expert_slice` 每个命中专家读取固定 `3·Es·H` 个元素，总量只依赖运行期命中专家数 `N_hit`，因此用本表输入 `N_hit`（默认未知），并在适配侧加相容守卫（见下）。
- 整网外围：Embedding/位置/mask/RoPE/main+compress/mHC 扩展/HyperHead 收拢/FinalNorm/logits 切片/LM Head 都有分类；`net_layer_loop` 是容器（成本 0 并写明“由层组串行汇总”）。mask 与两组 RoPE 频率在 `model.forward` 只构建一次（`modeling_deepseek_v4.py:1344-1362`）并传给全部层，故层组内为引用行、整网只在 `net_mask_build`/`net_rope_main`/`net_rope_compress` 收一次；`层组小计×层数`不会重复收这三项。

## 守卫与未知（不把未知折成 0）

- 维度：每个 B/M/K/N 的**符号本身与最终维度**都要求 `ISNUMBER ∧ INT(x)=x ∧ x>=0`，且 `repeat` 要求 ≥1 整数；负数/小数（如 H=-1、H=1.5）→ 维度/工作量/耗时/状态全部显示“未知（维度须为非负整数：…）”，**不产生负成本也不静默截断**。合法零维度（空缓存压缩条目、空窗口）保留 0。
- 吞吐：`ISNUMBER(吞吐) ∧ 吞吐>0` 才计成本；≤0、空白、文本 → 成本未知（工作量仍显示）。
- `N_hit` 相容守卫（DeepSeek 适配侧声明，通用导出器消费）：`N_hit ≤ min(E, B·seqlen·K)`；非空调用还需 `N_hit ≥ K`（TopK 组）或 `≥1`（Hash 组）。不满足 → 专家切片未知，相关层组小计与整网总计未知；**不给出不可实现的完整总计**。表中说明该单值是“各层统一命中数”假设，实际路由分布不由本工具推断。
- 泄漏/缺失：任何依赖未给定或非法 → `未知（依赖参数未给定或非法：…）`；小计用 `COUNT` 判定“全部为数值才 SUM”，不用 `IFERROR(...,0)`，也不用忽略文本的裸 `SUM`。

## 实际验证

1. **规则手算锚点**（独立求值器，不读 Excel；[rules-coverage.json](T10-cube-vec-verify/rules-coverage.json)）：CUBE B1/M1/K64/N64 = 1 ms、B2/M3/K64/N64 = 6 ms；VEC M1/N64 = 1 ms、M2/N64 = 2 ms、M3/N100 = 4.6875 ms（不向 64 取整）；真实 `q_a_proj` B1/S4096/K4096/N1024 = **4,194,304 ms**。全部通过。
2. **原生重算 + 改表**（Artifact Tool 导入交付工作簿、`recalculate()`、导出临时副本到 `/private/tmp/t10-cube-vec-lo`；[verify_workbook.py](T10-cube-vec-verify/verify_workbook.py) 只读核对，结果 [native-check.json](T10-cube-vec-verify/native-check.json)）：
   - **结构/公式 27/27 通过**：两页、5 参数区、5 算子区、27 列、每行都有分类、mask/RoPE 引用行与外围计费行、单层小计 `COUNT+SUM` 守卫、成本 6 位小数、串行 F/G/H 语义与“F 引用单层小计”。
   - **数值/守卫 35/35 通过**：默认 `q_a_proj`=8,192、`q_b_proj`=65,536、`attn_hc_comb`=2、`qk_scaled`=512、`attn_residual_mix`=128、`net_embed`=512、`net_mask_build`=1、`net_lm_head`=1,034,240 ms；seqlen=4096 → `q_a_proj`=4,194,304；吞吐 2048 → 全部成本恰好翻倍；吞吐 0/文本 → 未知；清空 seqlen → 未知、恢复 → 回到 8,192。
   - **命中数相容**：默认 `N_hit` 未给定 → 4 个层组各 1 项未定、整网总计未知（已知部分小计 1,042,065.625 ms）；`N_hit=6`（默认 1·8·6=48 分配，相容）→ 专家切片 2,359,296 ms、整网总计 129,612,141.5 ms；`N_hit=256` 在默认 48 分配下**不相容 → 未知**；`N_hit=0` → 未知；seqlen=4096 + `N_hit=256`（24,576 分配）→ 专家切片 100,663,296 ms、整网总计 **33,530,078,720 ms**。
   - **维度守卫**：H=-1 与 H=1.5 → `q_a_proj`/`shared_gate_up`/`net_lm_head` 等全部未知；全部 13 个副本的成本/工作量列**无负数**。
   - **公式链独立复算**：逐算子用 `scenario_evaluator` 从工作表里的参数/输入值独立重算规则，与 Excel 缓存值比对——**默认可求值 280 行、填 `N_hit=6` 后 284 行，两组各自全部一致**（含 `i_scorer` 两个矩阵项分行求和）；没有把“第一项”当整项。
   - **无公式错误字面量**：13 个副本全部扫描，`#`/`Err:` 命中 0。
   - **小数显示**：成本列 `#,##0.000000`，1/4096 ms = 0.000244 这类非零小值可见（[perf-first-ops.png](T10-cube-vec-verify/perf-first-ops.png) 可见 2,048.000000 等）。
3. **小例（单层×层数只乘一次）**：合成 fixture（单层 2 ms、3 层）断言 F 引用单层小计、G=3、H=F×G，整网 6 ms 而不是 18 ms（`test_serial_summary_multiplies_layer_count_once`）；原生副本中 `swa-hash` 的 H 也等于其单层小计×2。
4. **渲染**：Artifact Tool 只渲染新性能区与受影响页（[perf-first-ops.png](T10-cube-vec-verify/perf-first-ops.png)、[perf-group-totals.png](T10-cube-vec-verify/perf-group-totals.png)、[perf-serial-summary.png](T10-cube-vec-verify/perf-serial-summary.png)），无截字/`####`/错误值；串行说明行已横向合并避免窄列挤压。
5. **跨模型字节回归**（[cross-model-regression.json](T10-cube-vec-verify/cross-model-regression.json)）：用当前共享导出器从四模型**当前候选**结构重新导出工作簿，GLM-5.3 / GLM-5.3-Flash / Qwen3.8-Flash-Next / HiDream-O1 四本 xlsx **逐字节相同**（未启用性能区的模型不受影响）；DeepSeek 新旧候选的 `layer_structures.json`、`network_overview.json` 逐字节相同（结构语义未回退），只有工作簿按预期变化。
6. **候选闭包**（[candidate-closure.json](T10-cube-vec-verify/candidate-closure.json)）：新候选 `analysis_id` 由 `identity` 重算 = 目录名；7 个交付文件 SHA 与 `run_manifest` 一致；工具清单含 `deepseek_perf_rules.py`；**工具清单与冻结代码逐文件一致（drift 为空）**；旧候选 `61b74593…` 闭包一致、工作簿 SHA 仍为交付索引记录的 `2da0caf1…`。
7. **测试**（[tests-final.json](T10-cube-vec-verify/tests-final.json)）：全量 `discover` 2015 项，失败 6（`test_hf_decoder_diagram.py` 旧布局断言，T09 接手即登记）+ 错误 16（本机缺 onnx/onnxsim/onnxruntime，T09 已登记），跳过 20；**本票受影响的 13 个模块全部通过**，其中 `test_operator_workbook_v2.py` 从 T09 的 24 项增至 35 项，即**新增 11 个测试方法**（覆盖性能区/分类/维度守卫/`repeat` 守卫/命中数相容/串行只乘一次/整网外围排除）。`tests/hf/` 无 `__init__.py`，不在 `discover` 范围内，需显式运行（Codex 独立跑 18 模块清单 607 项时唯一失败就来自这里）：唯一因新代码暴露的静态守卫是 `tests/hf/test_operator_workbook.py` 的 stdlib 根白名单未含 `math`。已说明用途（合并说明行的行高换算 `math.ceil`，不读磁盘/进程/HF/torch）后**只扩展白名单**，守卫本身保留；该模块 57 项现全部通过。
8. **未修改无关命名或组件去凑全绿**：6 项 diagram 失败与 ONNX 缺依赖均按 T09 记录保留，不计为本票通过。
9. **Codex 独立复核**（[final-independent-check.json](T10-cube-vec-review/final-independent-check.json)，结论 `passed`）：独立重建 `dcbf7eae…` 并核对源/工具/7 个产物 SHA、291 个算子实例分类、其余四模型 xlsx 字节、同进程隔离、单层兼容、HTML（归一化 analysis_id 后）相等，以及 Prefill/Decode 两页 10 组原生 LibreOffice 重算；18 模块清单 607 项全部通过。

## 复用 / 重构 / 新增 / 删除

- **复用**：既有同源 `layer_structures.json`/`network_overview.json`、五模型唯一 `export/operator_workbook.py` 导出链、`hf_workbook_common.py` 的输入/参数/层组声明、`scenario_evaluator.py` 白名单编译与三态守卫、`build_hf_decoder.py` 的内容寻址身份与落盘、DeepSeek 四层组与整网外围的真实结构/来源/shape/循环归属。
- **重构**：`operator_workbook.py` 把列布局、横幅、网格合并、参数/算子区渲染参数化为 `layout`，新增可选性能区（列/维度/工作量/成本/状态/小计/串行表），未启用时逐字节保持旧输出；`hf_workbook_common.py` 的输入/说明/整网区支持性能引擎（默认关闭）；`hf_workbook.py` 给每个算子挂声明式规则并校验“算子全覆盖 + 符号都在本组声明”；`build_hf_decoder.py` 的 DeepSeek README 与工具身份清单更新；`cli.py` 帮助文案更新。
- **新增**：`export/deepseek_perf_rules.py`（291 个算子实例的 CUBE/VEC 规则、整网外围块与串行顺序、`N_hit` 相容守卫）；`T10-cube-vec-verify/` 的规则清单/原生重算/跨模型回归/候选闭包/测试汇总/渲染图脚本与结果；本报告。
- **删除**：无。旧候选、旧证据、旧单算子模板、根两项既有删除的状态均未改变。

## 限制与未验证

- 数值来自用户**归一化基准**（4096 MAC/ms、64 元素/ms）；量级很大（如默认整网总计 10^8–10^11 ms 级）是基准本身的结果，不是设备实测，也不代表真实硬件。默认 Prefill 因 `N_hit` 未给定而**没有完整总计**，只给已知部分小计与 4 个缺口；填相容 `N_hit`（如 6，或 seqlen=4096 时 256）后才给完整数值。
- 专家模板用“总路由分配数”和“各层统一命中数”表达，不做运行期分布推断；`N_hit` 单一输入是显式假设，不是路由器结论。
- 不计带宽、并行、融合、通信、精度/位宽折扣与真实设备 kernel；精度与位宽输入保留但不影响耗时；不声称反映零拷贝/真实算法复杂度（`hc_comb` 的 Sinkhorn 迭代、view/transpose、clamp 等不另收费均已注明）。
- 单层兼容入口 `tables/decoder_reading.xlsx` 也启用同一套规则（无整网外围块，mask/RoPE 按一次计入本层并注明）；它不是本次整网交付物。
- 未在 Microsoft Excel GUI 手工编辑；验证引擎是 Artifact Tool + 其原生重算。未做用户阅读验收。其余四模型只验证“工作簿字节不变”，**性能未接入**（T11 由 Codex 决定）。
- 共享导出器/公共声明源码已改：其余四模型**下次重建会得到新 `analysis_id`**，但工作簿字节不变（本轮已逐字节验证）；本票没有重建、覆盖或改写它们的当前候选，交付索引中其结构基线仍有效。

## 停止状态

完成本票实现与验证后停止，等待 Codex 复核。GitHub #11 保持 OPEN；未 commit/push；未改 `execution.json`、`tickets.json`、`monitor-state.json`、任务 README 或自动化配置；自动化仍 PAUSED/45 分钟；未派发或开始 T11。
