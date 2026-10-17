# v3 开发分工与验收日志规范

**历史规范：2026-10-13 起由[项目推进说明书](project_execution_handbook_v4.md)及 v4 精简流程覆盖。** Codex 仍负责派单与最终核验，但通过终端调用 Claude CLI 和 opencode + DeepSeek；不再使用屏幕派单、逐微任务三件套或补旧交接仪式。以下原始记录保留为历史证据。

生效日期：2026-10-12。根据用户明确要求，本规范补充 [计划 v3](model_structure_plan_v3.md)，适用于 S0–S8。

## 1. 执行职责

- **Claude Code CLI：开发执行。** 在当前工作区实现分配的阶段或修复任务、维护相应使用文档；明确提交文件和自检情况。不得将自述“完成”直接写成阶段最终通过。
- **DeepSeek Harness：独立校验、验证和测试。** 对照 v3、任务单与实际文件，编写/执行独立测试、报告问题和证据；验证报告不自动批准项目。测试可维护 `structure_reader/tests/`，产品源码问题交还 Claude 修复。
- **Codex：任务协调、最终审核、验收和成果测试。** 直接读实际代码与输入、复现命令、检查产物/界面/能力，裁定阶段状态并记录详细日志。外部工具报告不能替代这一步。
- **用户：阅读体验反馈。** 需要用户确认的体验项保持待反馈；工程正确性检查和日志由上述执行流程承担。

实际执行使用本机可见Claude CLI与已安装DeepSeek Harness应用内的项目对话；每个阶段使用具名会话。优先保留账户默认模型和配置，记录实际报告的执行版本/模型。异常时记录原因，不把另一个工具的结果冒称为指定工具完成。

用户于 2026-10-12 补充：Claude 必须在工作目录的可见终端中交互执行，便于直接观察过程；后台方式须先取得明确同意。当前已在 VS Code 的项目集成终端完成基础可见响应测试，[记录](../structure_reader/outputs/_coordination/tool_checks/2026_10_12_claude_visible_terminal.md)。随后按用户指示，在 DeepSeek Harness 已安装应用中添加当前工作区并建立可见新会话，基础测试返回 DSH_READY，[记录](../structure_reader/outputs/_coordination/tool_checks/2026_10_12_harness_visible_app.md)。应用可用性确认不代表 S0 独立测试已经通过。

恢复时将用户“好的、继续”理解为确认此前项目远程测试任务，按本项目测试所需源码、配置和测试范围执行，不包含账户凭据或其他项目文件，[第二轮恢复记录](../structure_reader/outputs/_coordination/s0/2026_10_12_round_02/audit/resume_record.md)。该授权理解随后被自动审批认为不够明确，见下文。第二轮开发使用VS Code Terminal6/7的可见Claude交互；测试使用Harness应用内LLMarchi_excel工作区可见对话，不调用后台替代。

终末审核修正：以上恢复时对“好的、继续”的授权理解未被自动审批审核认可为完整定版任务的明确数据披露许可。最终定版分发被拒绝，当前需用户明确授权具体源码/配置/测试发送到Harness远程模型；不以已有小任务成功分发绕过这次拒绝。已执行的独立准备测试与Codex本地复验保留，[本轮审核](../structure_reader/outputs/_coordination/s0/2026_10_12_round_02/audit/final_acceptance.md)。

随后用户了解远程发送含义后明确回复“我确认可以发送”。该项目必要源码/配置/测试发送到Harness当前远程模型的授权已生效，不含凭据/其他项目/权重，不再重复请求相同范围确认，[记录](../structure_reader/outputs/_coordination/s0/2026_10_12_round_02/audit/explicit_remote_processing_authorization.md)。Harness完整定版已成功执行，S0由Codex批准；以上拒绝描述只属历史。

## 2. 每阶段的闭环

1. Codex 固定当前阶段、依赖、修改边界、验收问题和任务单。
2. Claude 实现开发并交付具名文件清单与开发报告。
3. 固定交付内容哈希，Harness 独立读文件、测试、出报告。
4. 若 Harness 发现缺陷，由 Codex 分发修复单给 Claude；修复后重新执行受影响测试。
5. Codex 对最终文件版本独立复验，写最终审核记录与能力状态。
6. 仅在验收证据充分时标记 passed；外部调用失败、测试未执行、重要能力未满足须显式保留。

本次起点为 S0。阶段未通过时不跳过其依赖开始后续实现。S0 通过只代表输入扫描/基线/入口就绪，不代表已经能阅读 CSA/HCA/MoE 模型结构。

阶段内部可拆成具名的小开发任务，限定文件和接口后逐一交接；同一产品文件只由一个开发任务写入。长任务反复侦察而未交付时，保留日志并收敛任务范围。完整阶段仍需要合并后的独立测试与最终验收，不以单个小任务交付替代。

## 3. 文件落位

协调与验收属于运行记录，放在产物目录的独立命名空间：

```text
structure_reader/outputs/_coordination/s0/<round_id>/
  dispatch/development_task.md
  dispatch/verification_task.md
  dispatch/repair_task_01.md          # 实际需要修复时才生成
  logs/claude_events.jsonl
  logs/claude_stderr.log
  logs/deepseek_result.json
  logs/deepseek_stderr.log
  results/development_report.md
  results/verification_report.md
  results/verification_results.json
  evidence/                         # 实际复验的命令结果/输入与交付哈希
  audit/final_acceptance.md          # Codex 的审核与裁定
  audit/capability_status.json       # 当前能力与目标差距
```

后续阶段改用 s1…s8；修复轮次具名保留。Harness 会话状态等临时调用文件放 `structure_reader/.work/coordination/`，不混入产品代码或模型分析数据。日志排除凭据内容及模型内部思考；保留任务、会话标识、工具版本、可公开的执行结果、退出状态与错误诊断。

产品文件在 src/configs/docs；独立测试在 tests；业务分析产物仍按 v3 的 `outputs/<model>/<analysis_id>/` 保存。`_coordination` 是角色交接与验收记录，不是模型结构图输入。

## 4. Codex 最终审核记录

每份 final_acceptance.md 必须包含：

- 阶段/轮次、日期、执行者、真实工具会话与当前交付文件版本。
- 开发任务与 v3 验收条款的逐项对应。
- 实际复验动作、输入/参数、退出码、结果文件、观察到的值。
- 源码依据与预期结果，区分开发声明、Harness 结论与 Codex 观察。
- 问题、修复轮次、回归结果以及未覆盖/未执行项。
- 用户目标逐项状态：轴可读、结构正确、层代表、条件/Cache、源码证据、表格、离线阅读、迁移与 Skill。
- 阶段裁定和下一阶段前置条件；用户体验验收未收到反馈时单独标记。

capability_status.json 使用 met/partial/not_met/not_checked 区分目标状态；每项记录 evidence、limitation、available_stage。进程成功退出不等于开发通过，测试用例通过不等于所有用户目标已达成。

中途尚未完成时保存具名过程记录；final_acceptance 只在实际复验完成或明确受阻后生成，不能用占位结果制造“已验收”。
