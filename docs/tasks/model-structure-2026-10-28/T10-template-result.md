# T10 模板阶段结果：单算子性能空白模板（q_a_proj）

状态：**模板制作与验证完成，等待 Codex 复核**。未关闭 #11，未进入原整网成本计算/T11，未改协调或自动化文件，未 commit/push；自动化保持 PAUSED、45 分钟。

## 接手核对

- 工作区：`/Users/xiechonghang/Workspace/LLMarchi_excel`；分支 `main`；HEAD `947ae78f120279403ec0a4c539e99d3d8869e1d6`（本票前后未变，未产生提交）。
- 接手快照：`/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T10-template-baseline-20261029-xslwnajv/manifest.json`，捕获于 2026-10-29T17:26:48+08:00，`661` 文件，内容副本在 `files/`。
- 接手时工作树 `git status` 共 104 项；当前 105 项，唯一新增是 `?? outputs/`（本票交付目录）。既有修改、根 `CONTEXT.md` 与 `2026_10_17_workspace_structure_review.md` 的删除、全部正式候选均保留。
- 快照后协调流程更新了 `docs/tasks/model-structure-2026-10-28/` 下的 `README.md`、`execution.json`、`monitor-state.json`、`tickets.json`（内容为 T10 派单字段）。本票未写入这 4 个文件，也未调用 `gh`，保持只读。

## 交付物

- 正式工作簿：[`matmul_performance_template.xlsx`](../../../outputs/01a17dc8-T10-template-20261029/matmul_performance_template.xlsx)，SHA256 `24c3f23725eed1a94ba6bedd1ac5505d6d340a2cb5ebda12aba9a929c9b57320`，`8116` 字节，单工作表 `MatMul 性能模板`。
- 可重现生成器：[`build_matmul_template.mjs`](T10-template-verify/build_matmul_template.mjs)，用 Artifact Tool `@oai/artifact-tool` 制作；它是本票一次性模板生成器，不是新增产品 Workbook 导出链，也未改 `structure_reader` 源码。
- 生成方式：临时目录把 loader `node_modules` 软链到 `T10-template-verify/` 后运行 `node build_matmul_template.mjs`；软链已在交付前移除。生成器把已验收候选与官方 `config.json` 中核对过的事实写死为默认值（另由 `expected_from_source.py` 从这两处重算核对），运行时不解析模型源码。

## 选定算子与默认值

| 项 | 取值 |
| --- | --- |
| 模型 / 候选 | DeepSeek-V4-Flash-0731，T09 已验收候选 `61b74593da1e9fe4f84cd0384035a42b0c356c9063d26470c6ca5f9ebba3309d` |
| 层组 / 层 | `swa-hash`（layer 0，43 层主干第 0 层） |
| 算子 ID / 模块 | `q_a_proj` / `DeepseekV4Attention.q_a_proj` |
| 算子类型 | `linear`，无 bias，无激活 |
| 源码来源 | `models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json`（hidden_size=4096、q_lora_rank=1024、attention_bias=false）；`hflib/transformers/.../modeling_deepseek_v4.py:783` 为 `self.q_a_proj = nn.Linear(4096, 1024, bias=False)`，调用在 `:813` |
| 权重布局 | 存储 `[N, K] = [Rq, H] = [1024, 4096]`（PyTorch `nn.Linear` 的 `[out, in]`）；计算按 `Y = X·W^T`，即权重按 `[H, N] = [4096, 1024]` 参与乘加 |
| 默认规模 | B=1（样本）、seqlen=4096（token）、H=4096、N=Rq=1024、精度 BF16、每元素 2 字节 |
| 规模约定 | 1 次乘加（1 乘 + 1 加）计 2 FLOPs；`FLOPs = 2 × M × H × Rq`，不是设备耗时公式 |
| 默认结果 | M=4,096；MACs=17,179,869,184；FLOPs=34,359,738,368；34.359738368 GFLOP；权重 4,194,304 个元素；X/W/Y 数据大小 33.55 / 8.39 / 8.39 MB，合计 50.33 MB |

## 表格结构与填写位置

只有一个中文工作表，按 ① 输入 → ② 结构 → ③ 性能 → ④ 约定 → ⑤ 来源与限制排列：

- **① 输入参数（浅蓝可改）**：`C6` 批大小（样本）、`C7` seqlen（token）、`C8` H、`C9` N、`C10` 精度说明、`C11` 每元素字节。默认值来自 config，用户可直接改。
- **② 结构 shape 与计算规模（自动重算）**：`C15` 输入检查；`C16` M；`C17`/`C18`/`C20` 输入/权重/输出 shape；`C19` 计算布局；`C21` MACs；`C22` FLOPs；`C23` GFLOP；`C24` 权重元素数；`C25:C28` X/W/Y 数据大小及合计（单次读写基线，不是实测片外流量）。
- **③ 性能参数（留空，由你填写）**：`C32` XPU 矩阵性能（TFLOP/s）、`C33` 内存带宽（GB/s，可选）、`C34` 理论耗时（**ms**）。三格在正式表中都是真正的空单元格，没有 0、`待提供` 或猜测耗时。
- **④ 约定与单位**：2 FLOPs 约定、权重 `[N,K]` 存储与 `[K,N]` 计算、`GFLOP ÷ TFLOP/s = ms` 与 `MB ÷ GB/s = ms`、空白与无效约定、数据大小口径。
- **⑤ 来源与限制**：算子、config 字段、源码行、候选与层组；明确只覆盖该 Linear。

填写方式：改 `C6:C9`/`C11` → 看 `C15` 是否「有效」→ 在 `C32`/`C33` 填硬件能力 → 在 `C34` 直接键入 Excel 公式。E34 给出同类量纲示例 `=C23/C32`（GFLOP ÷ TFLOP/s = ms），只是提示，不占公式输入格。空白表示尚未提供，不等于 0。

## 对 Codex 三处收尾的处理

1. **批大小单位**：`D6` 由 `token` 改为 `样本`；`seqlen` 仍为 `token`。
2. **未知/无效守卫**：`C15` 为唯一输入检查，
   `=IF(IFERROR(AND(COUNT(C6:C9,C11)=5,MIN(C6:C9,C11)>=1,C6=TRUNC(C6),C7=TRUNC(C7),C8=TRUNC(C8),C9=TRUNC(C9),C11=TRUNC(C11)),FALSE),"有效","未知")`。
   清空、文本、0/负数、非整数都会显示「未知」；`C16:C18`、`C20:C28` 的每个派生值都以 `$C$15="有效"` 守卫，任一结构输入无效时这些派生值整块显示「未知」，不会静默变 0，也不会拼出残缺 shape。`C15` 在「未知」时显示浅红底、红色粗体。性能 `C32:C34` 不受守卫影响，仍真正空白。未增加全局校验表或隐藏校验页。
3. **数据大小措辞**：`C25:C27` 改名为「输入/权重/输出数据大小」，说明分别写「X/W 的字节数（单次读基线）」「Y 的字节数（单次写基线）」；`C28` 为「数据大小合计」，说明「三项字节数之和；非实际流量」。④ 新增「数据大小口径」：`C25:C28` 只是 X/W/Y 的数据大小与单次读写基线，不含缓存复用、额外中间量或对齐填充，也不等于实测片外流量。

## 实际验证

1. **制作链**：首次创作前成功运行 Skill 的 Artifact Tool marker；`build_matmul_template.mjs` 完成 `recalculate()`、`inspect()`、`render()` 和 `exportXlsx()`；inspect 输出见 [build-inspect.ndjson](T10-template-verify/build-inspect.ndjson)。
2. **结构与数值**：[source-and-expected.json](T10-template-verify/source-and-expected.json) 由 [expected_from_source.py](T10-template-verify/expected_from_source.py) 从已验收候选与 config 独立推导；正式工作簿缓存值由 [verify_workbook.py](T10-template-verify/verify_workbook.py) 逐项核对，24 项形式/公式检查通过。
3. **原生 LibreOffice 重算**：正式表重算副本的 12 项数值/空白/无错误检查通过；`C32:C34` 仍为空，错误字面量为 0。
4. **临时副本改输入**：把 B=2、seqlen=1000、H=2048、N=512、字节=4，并在 `C32` 填 `300`、`C34` 填 `=C23/C32`；原生重算得到 M=2000、MACs=2,097,152,000、GFLOP=4.194304、`C34=0.0139810133333333` ms，18 项检查通过，证明用户可自行在空白 ms 格填公式并随输入重算。
5. **守卫副本**（均由原生 LibreOffice 重算）：清空 `C6`、清空 `C9`、清空 `C11`、`C6="abc"`、`C9=0`、`C7=10.5` 六种情况，`C15` 均为「未知」，`C16/C17/C18/C20/C21/C23/C25/C28` 均为「未知」，性能三格仍空，无任何公式错误字面量；恢复 `C6=1`、`C9=1024`、`C11=2` 后重算回到默认值。14 项守卫检查通过。守卫视觉证据见 [native-render-guard-blank-batch.png](T10-template-verify/native-render-guard-blank-batch.png)。
6. **渲染**：原生 LibreOffice 渲染唯一工作表 `A1:E45` 共 1 页，[native-render.png](T10-template-verify/native-render.png)；Artifact Tool 渲染见 [render-artifact-tool.png](T10-template-verify/render-artifact-tool.png)。两种渲染均检查标题、单位、shape、空白性能格和说明，无截字、无 `####`、无公式错误。
7. **正式文件未变**：以上验证只读取正式 xlsx，交付前后 SHA256 均为 `24c3f2…b57320`。
8. **候选与工作树**：5 个当前候选共 39 个产物 SHA/字节全部匹配交付索引，见 [candidate-check.json](T10-template-verify/candidate-check.json)；索引文件 SHA256 `3975d0c973bdcde931470914d744f841ed6e7f4ee9738ed63dfefe656d3e7434`。工作树对比见 [baseline-diff.json](T10-template-verify/baseline-diff.json)：新增为本票输出/证据，以及 Codex 复核目录下的 `round-1-findings.md`、`independent-check.json`（非本票写入）；修改仅上述 4 个协调文件，删除 0；五模型源码与产物零改动。
9. **Codex 复核证据**：[round-1-findings.md](T10-template-review/round-1-findings.md) 记录初轮三项（与本票收尾一一对应）；[independent-check.json](T10-template-review/independent-check.json) 记录最终模板 SHA256 `24c3f2…b57320`、默认数值、性能三格空白、单工作表与无错误检查为 passed；最终验收结论仍由 Codex 给出。

## 复用 / 新增 / 删除

- **复用**：T09 已验收 DeepSeek `layer_structures.json`、官方 `config.json`、transformers 源码行与候选身份；只读取，不再解析、不新建第二套模型解析器；没有改动既有唯一 Workbook 导出链。
- **新增**：正式 xlsx；`T10-template-verify/` 下的生成器、输入检查/核验脚本、期望值、候选与基线核验、渲染图、inspect/native-check 证据；本报告。
- **删除**：无。未改 `structure_reader` 源码、五模型候选、`model_delivery_index.md`、协调文件或自动化配置。

## 未验证与限制

- 未在 Microsoft Excel GUI 中打开或手工编辑；验证引擎是 Artifact Tool 与已安装原生 LibreOffice，不能证明 Excel 的最终显示或所有版本行为。
- 没有实际设备性能、实测带宽或用户性能公式；性能三格空白，不代表 0，也不代表已给出耗时。整网成本范围仍待用户公式，本模板不冒充 T10 完成。
- `C25:C28` 只是 X/W/Y 的数据大小与单次读写基线，不含缓存复用、额外中间量、对齐填充、通信、并行或融合。
- 只覆盖 `q_a_proj` 一个算子、一个工作表；未做性能曲线、完整 roofline、整网汇总或其它模型。
- 未做用户阅读验收；等待 Codex 复核。

## 证据文件

- [build_matmul_template.mjs](T10-template-verify/build_matmul_template.mjs) — 模板生成器
- [expected_from_source.py](T10-template-verify/expected_from_source.py) / [source-and-expected.json](T10-template-verify/source-and-expected.json) — 算子来源与期望值
- [build-inspect.ndjson](T10-template-verify/build-inspect.ndjson) — Artifact Tool inspect
- [verify_edit_copy.mjs](T10-template-verify/verify_edit_copy.mjs) — 改输入/守卫/恢复副本
- [verify_workbook.py](T10-template-verify/verify_workbook.py) / [native-check.json](T10-template-verify/native-check.json) — 68 项核对（24 形式 + 12 原生默认 + 18 改值 + 14 守卫）
- [verify_candidates.py](T10-template-verify/verify_candidates.py) / [candidate-check.json](T10-template-verify/candidate-check.json) — 五候选未变
- [compare_baseline.py](T10-template-verify/compare_baseline.py) / [baseline-diff.json](T10-template-verify/baseline-diff.json) — 接手基线增量
- [native-render.png](T10-template-verify/native-render.png)、[native-render-guard-blank-batch.png](T10-template-verify/native-render-guard-blank-batch.png)、[render-artifact-tool.png](T10-template-verify/render-artifact-tool.png) — 渲染证据

## 停止状态

完成本模板阶段后停止，等待 Codex 复核。GitHub #11 保持打开；未 commit/push；未改 `execution.json`、`tickets.json`、`monitor-state.json`、任务 README 或自动化；自动化仍为 PAUSED/45 分钟；未进入整网成本/T11。
