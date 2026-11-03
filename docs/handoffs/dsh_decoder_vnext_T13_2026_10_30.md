# T13 正式交接：DeepSeek 产品分层浏览接入

仅在 Codex 已验收 T12 且 execution.json 当前实际派发为 T13 时执行。用户授权本版本仍由 dsh 开发，Codex 派单及验收；GitHub #14，完整要求 `docs/tasks/model-structure-2026-10-28/13.md`。

T12现已验收并关闭#13，验收入口 `docs/tasks/model-structure-2026-10-28/T12-decoder-review/acceptance.md`，最终数据为 `T12-decoder-verify/round-3/`。T13接手快照2026-10-30 20:31:26 +08:00，1158文件/14814历史产物/40当前候选，manifest：`/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T13-decoder-vnext-baseline-20261030-fy_4uifk/manifest.json`。快照后Codex对当前交接、execution/tickets/monitor/任务README等协调更新只读登记；独立复核目录也由Codex维护，保留T12全部历史证据。

## 接手

1. 在 `/Users/xiechonghang/Workspace/LLMarchi_excel` 核对实际 cwd、HEAD（预期947ae78f120279403ec0a4c539e99d3d8869e1d6）及 AGENTS.md 指向的领域资料。当前实现包含大量未提交内容，HEAD不是完整实现身份。
2. 读取 `docs/tasks/model-structure-2026-10-28/execution.json`、tickets.json 中 T13 与 T12 验收记录。以当前 execution 的 baseline_manifest 为唯一接手快照，逐文件复算哈希、保护历史产物/已验收工作簿、已有删除；把差异与快照后 Codex 协调更新区分并写入 T13-decoder-verify。无关非协调差异先说明。
3. 读 T13 任务、最新 `docs/requirements/decoder_dataflow_refinement_2026_10_30.md`、T12 最终验收记录/最终语义接口与预览。T12旧轮报告与旧候选仅为历史，使用验收指定的最终轮数据。

## 实施目标

让用户从实际 DeepSeek 产品入口按 整网→层组→Block/Decoder→Attention/KV→FFN/MoE/mHC/Norm/残差→返回 的路径阅读新版图。底层由 Archify 绘制，源于同一raw/normalized语义与来源，不把独立预览链接集合冒充产品集成。

复用 `build_hf_decoder.py`、HF事实/模型适配/来源校验、现有 `render/hf_structure_page.py` 路由与详情、T12语义层与Archify适配。先查真实调用方，再扩展必要接口；不能为各视图手抄模型结构或另造构建器。复用要在报告中指出具体接口；被明确替代的重复代码先查引用再处理。

完整 Decoder 的具名计算必须能读出：Linear保留真实实例名；FFN/MoE内部Linear、路由/分派/单专家模板/共享专家/汇合按需进入，运行期专家命中与每专家token数仍未知；mHC的collapsed/post/comb与残差是真实数据依赖，不能只贴说明。Norm/RoPE内部微步骤保持折叠，不实例化模型/加载权重。

主链沿拓扑方向单向展开，全称维度与真实张量身份贯穿；所有分支/共享消费者都保留。辅助cos/sin、mask、历史缓存以靠近消费者的短支路接入，同一张量引用需可联查来源；历史输入、更新后状态输出不得制造回流。不得靠省略边/微小字体/无依据合并换取整齐。raw/normalized和具体规则应用/拒绝、条件/未知与源码详情可按需查看；超过摘要数量的关系也要有可点击入口。

离线页面、关联子页与返回上下文可用，不依赖一次性HTTP服务。保留四变体/43层归属、单层兼容、来源拒绝、其余四模型原入口。现有五模型Excel逐字节保持，CUBE/VEC与收费/汇总本票不动；图节点拆分与旧表对应清楚。

新候选身份覆盖最终语义/规则/renderer/模板/工具包依赖与来源哈希；不要把未提交实现当作根HEAD已提交代码。Archify按技能运行finalize，真实几何失败必须修正，环境缺失如实单列；不修改安装技能包、不硬编码个人路径。工具身份与最终HTML/浏览器证据必须绑定同一轮产物。

## 验证及停止点

使用新候选目录，保留历史产物与T12最终证据。完成真实构建、必要定向回归、五模型旧工作簿保护，实际浏览器走通任务单中的阅读路径，检查所有真实连接方向/分支、全称形状、点击详情与返回、离线链接和不同变体，查看渲染图。不要用节点计数或只有JSON字段存在代替用户操作验收。

若本会话Chrome因启动权限不可用，准确记录并交可重复步骤及最终产物，Codex可在可用环境补查；任何自动失败都不能伪称通过，Codex最终验收前浏览器要求必须完成。

报告 `docs/tasks/model-structure-2026-10-28/T13-decoder-result.md`，证据 `T13-decoder-verify/`：新候选入口、raw/normalized与来源/节点映射、导航/路线/视觉证据、可复现身份、测试、旧产物保护、复用/扩展/新增及限制。

完成T13即停止等待Codex，不提前接T14、不关闭Issue、不commit/push。协调文件 execution/tickets/monitor-state/任务README/executions/由Codex维护，本票只读；不修改自动化、不恢复旧会话、不归档聊天。你的报告和证据可写，已有脏改动/删除/模板与旧候选保持。
