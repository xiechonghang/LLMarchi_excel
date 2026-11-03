# D03 复核：退回返修

日期：2026-11-03（Asia/Shanghai）。候选 `7c724c14554c404fc838785a956819f85a0f3d8039a353d9a9792f981274fe01`，正式页 SHA256 `fd107e6d1b8cc3cfeee58101acbd60ee38b90a69291ee7886b240bf3241cb4db`。D03/#19 未验收，不关闭、不启动D04/D05。

## 已确认

- 独立阅读8个变化文件，补跑阅读页面测试45/45通过。551项为dsh全套报告，不冒充Codex全跑。
- 42个真实图位均被既有探针动态遍历，两种视口共514项：496通过、18失败、0运行错误，120张截图。18项中2项仅为旧探针硬编码“33图位”，实际42项是预期新增，不计产品缺陷；其余16项为8张新图×2视口的真实字号失败。
- Q/KV并行、Indexer接q_residual、TopK不物理裁剪KV的修改方向保留。D01/D02入口、整网与mHC职责不重做。
- 22641既有输出文件全部未变，Excel与D02候选逐字节相同，实际来源清单与当前源码一致。2827协调文件中仅tickets.json有Codex派发时的外部变化。详 `D03-codex-verify/protection.json`。

## 阻塞项

1. **P1 Attention实际交付图丢失block_bias真实输入。** `attn.block_bias`被标成`kind=aux`，`build_view_model`按只读辅助源投影成 `n-attn-block_bias-attn-mask_extend` 外部输入，仅留下它到mask_extend的边。CSA最终candidate无compressor→block_bias、Indexer→block_bias；HCA同样缺压缩条目→block_bias。`test_archify_preview.py`新增expected减去`e.compressor.block_bias2`/`e.indexer.block_bias`，掩盖了实际投影丢边，不能采纳。要求真实计算节点及输入输出保留在最终candidate/relations，而非只检查语义bundle；恢复完整trace断言并覆盖最终公开产物。

2. **P1 HCA偏置节点被错误套用CSA且细图遗漏实际算子。** `deepseek_v4_semantic.py`对所有has_compressor生成`DeepseekV4CSACompressor.block_bias`，条件固定`layer_type == 'compressed_sparse_attention'`、参数含index_topk，输入没有position_ids。HCA源码391–440按position_ids与compress_rate构建因果可见性，seq_len==1或compressed_len==0时返回None，没有Indexer/TopK。实际HCA facts含`hca_block_bias`，而`build_compressor_graph`只取`c_*`，因此HCA细图遗漏该算子，却包含其后`c_mask_extend`并从段外接入block_bias。修正真实类型、条件、必要位置输入及None分支；细图完整覆盖实际边界，不把HCA伪装CSA。不要求机械展开所有微算子。

3. **P2 缓存主图隐藏状态身份/维度并缺少更新输入，就近参数不完整。** CSA缓存图主图只写“6项：entry_count/buffer/compressed/overlap（全量见关系JSON）”，汇总边shape取第一个tensor，其他5项的身份/shape不可见；main-in还把调用前后状态一起计入摘要。`cache.update`只有旧状态输入，没有本调用新KV与本次可读full的消费者/边界，主KV输出计数也不含累计长度更新。AC03/11/13要求读清调用前后、主KV、压缩条目与缓冲余量，关系members不能替代主图。允许合理分组/下钻，但每项必须在可达实际图中具名呈现完整符号维度，补新输入/可读输出和相应更新规则。cache/compressor/indexer仍通用block前缀参数卡含o_groups等无关参数，Indexer index_n_heads/index_head_dim/index_topk未就近呈现；改用各图实际参数与来源。

4. **P2 新图原生及嵌入实际投影字号失败。** 1440视口：CSA cache约3.760px、CSA compressor约4.356px、CSA Indexer约4.159px，HCA cache约5.561px，HCA compressor约2.395px；两CSA组与1920视口同类失败，共16项。共享布局`wide_d03`将节点宽需求×1.8、上限2000，HCA原生viewBox宽4616，导致整体缩小。独立原生HCA browser-check确认Chrome available、`viewer/projected-text-readability`失败，1440下2.333px；这是产品字号问题，不是环境。修复真实布局/适当分层/原生阅读方式，保完整文字/边，不降6px门槛、不隐藏标签/无限扩画布。证据 `D03-codex-verify/page-authorized/result.json` 与 `native-hca-compressor/compressor-hca-topk.browser-check.json`。

5. **P2 Q/O缺独立可达通路入口。** 当前内部页签只有Attention总览/Q·KV·O整张图与KV新投影，没有单独Q、O入口。保留关联总览，同时补需求规定的分别进入Q/KV/O实际通路；不能仅改标题称已交付。各通路保持具名投影/Norm/轴变化/部分与逆RoPE及分组输出，节点到实际通路的入口/返回可离线使用。

## 验证范围与限制

本轮未逐张目检长图；按用户偏好由用户负责视觉。机器字号失败与技术丢边先修，暂不请用户重复审查此候选。修复后提供精简新增/变化入口。

本地受限首次Chrome启动失败，后续授权环境在保留Chrome sandbox下完成真实探针与一次原生HCA门禁，不能继续写全部环境失败。旧探针硬编码33图位需在新的D03探针副本中按真实manifest/42图位契约更新，保留历史探针与阈值。它实际动态遍历全部42图位，dsh runbook关于只遍历旧图位的说法不准确。

发现阻塞后未完成38图全套原生独立重建；受限环境尝试已停止，原始日志保留，不宣称38张浏览器通过。已保存源码diff及最终图连接清单 `D03-codex-verify/source.diff`、`delivery-semantic-review.json`。

只返修D03，保留已通过部分与全部旧候选，完成停待Codex技术复核及用户视觉审查。
