# T14 正式交接：四模型推广与五模型 vNext 交付

用户已授权本版本实际开发仍交给 dsh。仅当 T13 已由 Codex 验收、GitHub #14 关闭且 execution.json 实际派发为 T14 时执行；GitHub #15，完整验收以 `docs/tasks/model-structure-2026-10-28/14.md` 为准。

派发基线：T13已验收并关闭#14，验收 `docs/tasks/model-structure-2026-10-28/T13-decoder-review/acceptance.md`；最终DeepSeek候选 `19735c597e42a48da31384fc17cf8821757a81d49ddb0b2bd49c6f94bbab1393`。T14快照2026-10-31 12:03:48 +08:00，1344文件、15227受保护历史产物、40个T11当前候选文件；manifest为 `/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T14-decoder-vnext-baseline-20261031-92m3amlp/manifest.json`。快照后Codex对本交接及execution/tickets/monitor/任务README/executions等更新只读登记。

## 接手

在 `/Users/xiechonghang/Workspace/LLMarchi_excel` 核对实际 cwd 与 HEAD（预期947ae78f120279403ec0a4c539e99d3d8869e1d6），读取AGENTS.md及其领域指针。当前实现包含大量未提交内容；HEAD不是当前完整实现身份。

以 `docs/tasks/model-structure-2026-10-28/execution.json` 中 baseline_manifest 为唯一实际接手快照，逐文件复算并登记差异，保护既有删除、所有历史候选、五模型已验收工作簿与单算子模板、T12/T13全部证据。快照后Codex协调文件更新只读登记；无关非协调差异先说明。确认T13最终验收/最终候选，历史未验收候选不能作新基线。

## 实施顺序和复用

先读任务14.md、最新 `docs/requirements/decoder_dataflow_refinement_2026_10_30.md`、T13验收以及 `structure_reader/docs/model_delivery_index.md`。检查现有 model_registry、各模型规则/提取、产品profile/路由、构建身份、T12通用语义/化简、T13 Archify产品集成接口与相关测试；说明哪些直接复用、扩展、新增后实施。使用 Archify 绘制底层图，不修改安装技能包。

四模型键：`glm_5_3_hf`、`glm_5_3_flash_hf`、`qwen_3_8_flash_next_hf`、`hidream_o1_image_hf`。保持各自实际来源/config与已验收范围：GLM共享/独立indexer；Flash KDA/DSA；Qwen GDN/QSA/PLE与门控残差；HiDream沿T08既定单次生成网络/UiT及已纳入的条件ViT、DeepStack路径。不要强套DeepSeek结构、擅自补不存在的缓存或把生成网络换成语言logits主路径；不新增原范围外的图像输入支路/外部采样调度。

每完成一个模型，把可查看入口、证据、已知限制和下一步更新到T14结果报告，再继续本票其余模型。阶段结果不等于整票验收；若遇真实上下文/工具阻塞，落盘当前模型与可恢复步骤供Codex续派，不隐瞒未完成部分。

沿统一事实/语义/renderer/构建管线声明模型差异；可为实际不同的模型增加适配声明/语义规则，不复制整套构建器或为每张图手抄事实。先逐模型交可查看候选，最后同进程构建五模型并核对互不污染、未知键/不支持组合具名拒绝。

每模型维持原真实层组/循环次数/分支/缓存前态后态和运行期未知量。原始/化简图、每条实际规则的应用或拒绝原因、同源节点/张量/源码对应可访问；不必强行应用没有证明的规则。Linear使用真实实例或功能投影参数身份（F.linear不能虚构nn.Linear实例），Norm/RoPE/Softmax内部微操作默认折叠。各新中间张量有同源维度，全称映射覆盖主图、权重、边、详情；未知显式说明，不显示空shape或泄漏B/S/H/Es等旧别名。

图接到每个模型自己的产品层级入口，维持整网/层组/Block/子模块与返回上下文；HiDream用实际网络层级。主链/分支按真实拓扑下行，历史状态输入与新状态输出分开。粗模块收缩产生环时拆阶段，不能用SCC同秩隐藏双向回流；辅助cos/sin/mask等用局部短旁枝，完整关系/来源详情有可点击入口。避免微小字体、无效空栏及无限展开。

## 来源、身份和交付保护

每模型实际源码来源、仓库/revision与文件哈希闭合，复用既有source fingerprint拒绝逻辑。不要用根HEAD冒充未提交实现或嵌套来源，不伪造Archify repo证据。HiDream官方实现固定来源与其他模型不同，使用其实际已验收检出身份；必要工具参数可配置，不硬编码个人路径。

新候选覆盖最终事实/语义/规则/renderer/模板/包装页/关系数据及实际工具包依赖哈希，离线图和返回可用。运行门禁、Chrome可用性、时间戳/回执与稳定交付分离；同输入重建幂等。保留全部旧候选，不删除草稿或历史来掩盖错误。

五模型Excel与T11已验收工作簿逐字节保持，CUBE/VEC公式/收费/汇总不变；图的新粒度与旧表对应明确，不新增计费。用户后续再讨论Excel内部计算，本票只改结构交付。

## 验证、报告与停止点

针对真实边界做有意义的测试与相关回归（维度/分支/缓存/来源拒绝/跨模型状态污染/首次构建/稳定身份），历史diagram失败与ONNX缺依赖单列，不当作通过，不反复跑昂贵全量。每个最终候选做真实Archify门禁/浏览器阅读路径并查看代表渲染图，具体点击子图、详情和返回；证据绑定最终字节。环境失败如实说明，真实几何/语义失败修复，不擅自绕过安全策略或改安装工具。

最终更新 `structure_reader/docs/model_delivery_index.md`，统一五模型当前新版入口、模型范围/未知量、raw/normalized/规则/来源映射和旧表对应。完整报告 `docs/tasks/model-structure-2026-10-28/T14-decoder-result.md`，证据 `T14-decoder-verify/`；提供逐模型阶段入口与最终验证/产物身份/限制/保护清单。

完成仅T14后停止等待Codex。不要自行关闭Issue、宣布验收通过、commit/push、改自动化或恢复旧会话。execution/tickets/monitor-state/任务README/executions及Codex复核目录由Codex维护，本票只读；你的报告/证据与产品交付索引可写。
