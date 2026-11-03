# M2 ONNX完整维度名反馈最终复核

2026-10-25。用户截图反馈ONNX参数仍不合预期，当前确定缺陷为旧570b文件未同步此前完整维度名要求。Codex实际核512=head_dim、64=qk_rope_head_dim、32=RoPE频率半维，与根config/已核数学对应；Mul标准schema的第二输入端口名B不等于batch符号，不改标准算子。用户如另指具体数值/shape错误，仍可补充，不据本次命名修复代签其他数值问题。

原DeepSeek Harness App1017 High第41轮正常完成空闲。新候选`6126726e5bc849e939bfa00e56d7c398719568850f3110328e3f59b4ad001f43`：[HTML入口](../../outputs/deepseek_v4_flash_0731_hf/6126726e5bc849e939bfa00e56d7c398719568850f3110328e3f59b4ad001f43/viewer/decoder_variants.html)。仍四组完整Prefill→Decode与两表/原简ONNX/本地Netron；并未开启M3。

产品变化仅onnx_module/build.py：模型input/output/value_info的dim_param与state输出shape复用完整名映射，sim后再次投影，unk__N诚实标为unresolved_dimension_N；不猜未知维、不修改实际dim_value或计算图。新增两个有限字段投影/幂等检查；fixture仅用于protobuf字段断言，不冒充数值或shape正确性证明。原图/JSON/Excel/真实源码/启动器保持。

Codex实际复核证据`.work/coordination/hf_m2_onnx_names01/`：

- 读实际产品diff/两小测试/真实子进程runner，复跑2项通过。已有源范围比较只有build.py发生产品变化；需求/计划/progress是Codex按本次反馈授权更新，旧607冻结/报告保持只读，不称其意外产品漂移。
- `verify.py`直接读全部276原始与简化ModelProto，与570b每个模型逐节点序列/属性/引用、initializer字节、输入输出tensor名/dtype/静态尺寸逐项相同；共15126处符号轴标注换为完整名。已有value_info逐轴检查，未知名一一对应且未伪造具体维度，所有新value_info无已知缩写token残留。真实模型checker通过，state简化returned true，输出shape完整名与json_id保留。
- 截图对应CSA主KV样本实际attn_normed=[batch_size,sequence_length,hidden_size]、kv_weight=[head_dim,hidden_size]、kv_norm_weight=[head_dim]；position_ids仍保原['1',sequence_length]标注类型，本轮不把既有符号数字字符串擅自重定义为数值维。
- reading_variants.json和xlsx逐字节同570b，142个SVG及源码pre整块同；138条新链接均实际binary --resolve-only成功，未改注册/权限/Swift二进制。
- 28工具源/7输入/全部交付SHA与新身份复算吻合；10851旧候选文件复算不变。result.md/result.json/全部构建尝试和历史均保留。
- 保存runner实际subprocess argv/cwd/executable/必要env，build3/build4真实rc0，stderr空，同新ID，首轮reused=false、次轮true；codex_scope_builds.json与原始logs可回放。

`onnx_dimension_names_static_engineering=passed`；真实Netron GUI/browser/native Excel未核、用户阅读pending。本轮只关闭已确定名称遗漏；ONNX仍无真实权重的FP32阅读特化，未主张HF全模型数值等价。标准算子A/B输入端口保持，真实数值与未知维性质保持，未绕过原浏览器file/协议限制。

自动检查仍PAUSED，M3未启，未Git提交或推送。原Harness已完成等待，无新反馈或实际错误不重复派单。
