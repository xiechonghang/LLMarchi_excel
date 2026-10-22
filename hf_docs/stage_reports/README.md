# 项目阶段性记录

这里保存项目计划、实际进度、验收证据和下一步安排的阶段快照。

**新对话先读[项目推进说明书](../project_execution_handbook_v4.md)**：最新分工、实际快照、V1–V3 工作包及 CLI 接续提示。下面的 v3 快照与分工描述为历史，不能恢复旧逐单流程。

- [2026-10-13：v3 项目阶段整理](2026_10_13_v3_project_checkpoint.md)：S0已批准，S1局部IR和检查器已取得进展，完整S1与模型阅读页尚未验收。
- [原定执行计划 v3](../model_structure_plan_v3.md)：项目范围、接口与S0–S8验收标准的正式依据。
- [开发执行规范](../development_workflow.md)：Claude开发、Harness独立测试、Codex审核的职责约定。
- [工程进度入口](../../structure_reader/docs/progress.md)：持续更新的阶段状态。

## 保存约定

阶段快照使用 `YYYY_MM_DD_v3_project_checkpoint.md`；同一天需要新增快照时增加明确轮次后缀。后续快照新增文件并更新本索引，保留历史记录。

本目录只放阶段性说明。产品代码在 `structure_reader/src/`，入口在 `scripts/`，独立测试在 `tests/`，运行交接与审核证据在 `structure_reader/outputs/_coordination/`，临时文件在 `.work/`。阶段快照不充当模型图、测试期望数据或正式通过裁定。
