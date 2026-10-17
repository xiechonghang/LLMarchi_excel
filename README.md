# HF 模型资料与结构阅读工作区

当前任务：建设以 DeepSeek-V4-Flash-0731 为主案例的静态结构阅读工具。[项目推进说明书](hf_docs/project_execution_handbook_v4.md)是新对话接续入口：v3 管产品需求，v4 管垂直切片，说明书管最新分工与恢复步骤。

执行分工：Codex 通过终端派单并最终核验；Claude CLI 开发；opencode CLI + DeepSeek V4.1 Flash 独立测试。保留里程碑门禁，取消逐微任务交接仪式。

当前快照（2026-10-13）：S0 历史已批准；V1 准备代码已有、尚未启动新执行链；本地 892 项回归中 1 failure、2 errors；模型结构阅读页仍未交付。下一包是修复规则/求值器红项，随后建立 CSA·TopK 真实阅读闭环。历史阶段快照保存在[阶段性记录目录](hf_docs/stage_reports/README.md)。

## 当前文件放在哪里

- `hf_docs/`：HF 资料、v3 需求、v4 执行计划、推进说明书和方法说明；旧计划/审查记录放在 `hf_docs/archive/`。[资料入口](hf_docs/README.md)。
- `models/`：原始模型资料、配置、分词、参考源码及权重输入。[模型目录说明](models/readme.md)。
- `hflib/`：第三方软件源码输入，包括 Transformers、Diffusers。[软件目录说明](hflib/readme.md)。
- `scripts/`：现有工作区级工具。[模型下载脚本](scripts/download_hf_model.py)。

## 工程与产物位置

**S0已通过Codex验收。** 用户明确授权后，Harness完成定版133/133及进度契约回归137/137；Codex核心/产物28项与137项本地回归通过。注册、inspect、固定基线和使用文档均已交付，后续按推进说明书进入 V1，不再补旧逐单闭环。[本轮最终审核](structure_reader/outputs/_coordination/s0/2026_10_12_round_02/audit/final_acceptance.md)、[能力状态](structure_reader/outputs/_coordination/s0/2026_10_12_round_02/audit/capability_status.json)和[工程使用指南](structure_reader/docs/user_guide.md)记录实际情况；首轮失败与审批阻断保留历史。各目录职责如下：

- `structure_reader/scripts/read_model_structure.py`：统一用户命令入口。
- `structure_reader/src/`：提取、核验、渲染与导出库代码。
- `structure_reader/configs/`：分析设置、模型注册与已审查基线。
- `structure_reader/tests/`：验收测试和独立期望数据。
- `structure_reader/docs/user_guide.md`：使用方式；其他 docs 分别介绍开发与输出字段。
- `structure_reader/outputs/`：正式阅读页、数据、报告和表格；每次分析有独立目录与交付导航。
- `structure_reader/.work/`：可丢弃暂存，不作为输入或交付。

当前尚无已验收的模型结构阅读页。已落盘模块和协调/审核过程记录位于工程目录；命令与模型结构产物按阶段实现和验收后提供，路径和接口以 v3 为准。
