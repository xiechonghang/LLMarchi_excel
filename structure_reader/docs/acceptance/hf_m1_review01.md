# HF M1 首次主干交付复核

日期：2026-10-17。结论：**需修复，未通过M1工程验收**。当前仅layer_idx=4的CSA+TopK样板，不是全模型交付。浏览器视觉、原生Excel及用户阅读均未通过；尚未派正式opencode产品验证。

Claude在原可见Terminal20正常done5:59PM，交付候选`d4a8e60be16a2fc214d4d87da569f1ef62eec2767f6cebb62c4b33aba20a16b8`。候选保留只读，位于`structure_reader/outputs/deepseek_v4_flash_0731_hf/`下对应内容ID目录。

Codex实际复现：25项HF自测、47项Excel测试、98项独立源码准备均原始rc0；两次真实build-hf均rc0、同ID且reused=true。7份实际产物与当前build字节一致；manifest所列6份产物SHA及11项tool_sources匹配；工作簿可回读，仅Prefill/Decode两表。独立98项属于源码期望准备，不是新产品正式验证。

这些检查未能证明实际结构正确，源码核对发现：

- KV重排、Attention输出到残差、MoE输入reshape等实际producer断开；专家模板自读自产中间量；mHC重复生产门控张量、孤立transpose的输出rank与真实transpose不符。不能用说明文字代替数据连接。
- CSA/Indexer的gate与overlap状态数据流不完整，Indexer读取未经归一的q_a；block_bias排列在其topk依赖之前。
- Attention概率序列轴误用Nkv+N_comp，AV输入维度不闭合；topk实际min长度、cache持久短历史长度、RoPE和Attention布局需校正；缺q_a_norm实际权重。
- 参数联查56个不同HTML目标缺失，共952个链接实例。HTML两阶段连续展示已保留，但联查尚不可用。
- 把AV概率输入换成q_a，原始语义检查返回零问题；临时输出根调用真实CLI返回rc0并生成HTML。仅修改临时HF源码副本的mHC pre由sigmoid变cos，或使用未知model key，同样错误成功发布。真实输入与已有候选未修改，坏产物只在临时根，未进入真实候选目录。

复核证据：`.work/coordination/hf_m1/codex_review01/`的`result.json`、`flow.json`、`wrong_av_input.json`、`source_probes.json`、两个探针脚本及原始命令元数据/stdout/stderr。该目录为详细诊断补充；本文件是Git跟踪的正式结论。

已在同一可见Claude会话下发`claude_fix01.md`并确认Improvising处理中；不clear、不换模型/账户/权限、不重发原任务。修复以真实数据流和必要静态语义为目标，保留正确配置解释、pi导出器与已交付成果，不恢复旧B4、长度矩阵或全边变异门禁。预期`claude_fix_result01.md`与新的内容寻址候选。pi与opencode维持已交付后的空闲；M1修复并复核后才正式独立验证，用户阅读后才M2。
