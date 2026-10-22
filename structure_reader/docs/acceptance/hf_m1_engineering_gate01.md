# HF M1固定样板独立工程验证

2026-10-17，Codex复核DeepSeek正式独立验证round01。结论：**engineering_gate=passed**，仅限固定layer4 CSA+TopK MoE+双mHC的Prefill/Decode静态结构阅读样板；**user_reading=not_passed**，M2/M3尚未启动。

候选：`29d70d15971b524bc41cf334fa3768ab851186f68005549628ea638dd51abbd4`。联合freeze SHA256：`3c0e7d2c7b0bad6bd8f4f20050d7a230e30b2f9177e942d576ced0dc382c05ca`。DeepSeek开始/结束75冻结文件及21旧候选文件一致；Codex再次逐项复算全部一致，新增7个独立检查文件SHA与交付一致。

独立检查以本地HF源码和根config为依据核对参数、mHC comb.T方向、Q/norm和共享KV、CSA/Indexer状态与新增长度、单专家二维权重片/双索引/加权一次。此前4个manual为旧检查器角色识别局限，经实际归一化Q+RoPE引用和专家拆行连接核实；18个warning裁定为复合算子中间声明或源码忽略的router_logits，不作为缺计算。独立自动检查存在有界启发式，不能推论为通用任意图验证器，也不证明数值等价。

Codex阅读新增检查及正负例，并实际复跑24项自包含检查和固定候选JSON/HTML/xlsx检查，均原始rc0。核对DeepSeek原始日志和两次真实build记录：同固定ID、第二次reused，现行7份字节与候选一致；wrong_av、临时HF源码sigmoid改cos、未知模型键三个隔离CLI均rc1/0成功HTML。现有44/57/121项自测与准备检查仅记录，不替代独立源码核查。

HTML静态顺序为完整Prefill后完整Decode，1112页内链接无缺失目标；xlsx仅Prefill/Decode且与同一JSON一致。浏览器实际渲染/交互仍blocked/not_verified，原生Excel重算/视觉not_verified；回读和静态检查不能代签用户阅读。页面仍有冗余中间张量声明，是否影响理解由用户样板阅读反馈决定，不追加清理框架。验证报告将moe_dispatch的ffn_flat称多余输入；源码对应范围包含zeros_like(hidden_states)，Codex将其保留为初始化上下文，不把它解释为one_hot打分输入。

独立原始报告：`.work/coordination/hf_m1/gate/round01/independent_result.md`、`result.json`及logs；Codex复核：`hf_m1/codex_gate_review01/result.json`和复跑原始日志。历史报告均保留只读。

用户阅读入口：[离线HTML](../../outputs/deepseek_v4_flash_0731_hf/29d70d15971b524bc41cf334fa3768ab851186f68005549628ea638dd51abbd4/viewer/decoder.html)、[两表Excel](../../outputs/deepseek_v4_flash_0731_hf/29d70d15971b524bc41cf334fa3768ab851186f68005549628ea638dd51abbd4/tables/decoder_reading.xlsx)。当前三个OpenCode角色均停止等待，不重派、不扩展M2/M3；先收集阅读反馈。
