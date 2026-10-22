# 模型结构阅读工作区

当前目标：以Hugging Face Transformers模型源码和对应HF `config.json`为依据，生成能够读懂算子连接、参数、维度及KVCache的离线HTML与Excel。逐模型适配，Skill作为入口。

当前先做DeepSeek HF decoder阅读样板：Attention（含KVCache）、MoE和mHC；完整Prefill在前、完整Decode在后。Excel只有两张自包含工作表。最终补齐配置实际使用的结构变体和粗粒度整网至logits；不分析训练、采样或设备优化执行。

## 当前接续入口

1. [已确认需求](structure_reader/docs/requirements_alignment_2026_10_17.md)。
2. [既有代码复用核查](structure_reader/docs/hf_reuse_audit_2026_10_17.md)：直接保留、适配与退出当前范围的资产。
3. [实施计划v5](hf_docs/project_execution_plan_v5.md)：替换旧V1/V2/V3和B4主线。
4. [最新实际进度](structure_reader/docs/progress.md)：只读顶部当前状态，旧条目保留历史。

## 现在实际有什么

既有工程已实现参考推理源码的输入扫描、结构图、离线页面、CSV和DSpark集成；这些成果的依据是模型仓库的`inference/model.py`，HF源码此前只是对照输入。**它们不能直接当作新需求的HF实现结果。**

当前已完成新需求确认、代码复用核查和重规划；新HF提取器、双阶段页面、两表.xlsx和Skill尚未交付。下一包为真实HF decoder样板闭环，不再先补旧DSpark门禁。已有命令及历史能力见[工程README](structure_reader/README.md)。

## 目录与版本保护

- `structure_reader/src/`、`tests/`、`configs/`、`scripts/`：产品、测试、输入注册及入口。
- `structure_reader/docs/`：当前需求、复用核查、进度与工程说明。
- `hf_docs/`：新计划及背景资料；v3/v4和旧说明书已标为历史。
- `models/`、`hflib/`：只读模型资料和第三方源码输入，Git忽略，禁止用修改输入修工程。
- `structure_reader/outputs/`：生成产物与历史候选，Git忽略，原有内容保留。
- `structure_reader/.work/`：工作暂存与历史协调记录，Git忽略；目前含唯一历史验收证据，不能整体当成可丢弃目录清理。

已建立本地Git跟踪源码、测试、配置、文档；未配置远程仓库。本轮重规划不删除旧代码、候选或证据，不恢复用户已删除的根文档。新正式验收结论按v5保存为可跟踪文档，不再只放在`.work`。
