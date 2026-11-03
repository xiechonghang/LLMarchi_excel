# E03独立复核：正式接入通过

日期：2026-11-03。用户此前已确认layout-v2-rework布局并授权主工作区接入；本次复核只针对E03集成。固定差异依据E03-verify/pre-e03与E03-source-delta.patch，以及patch-apply中4个新增Excel文件，不将既有dirty tree归为本票修改。

## Spec轴：通过

- 主工作区DeepSeek all-variants构建已正式生成 `tables/decoder_performance.xlsx`；代码直接消费本次真实structures/network，无E01固定输入旁路。
- Codex独立运行推荐CLI的JSON模式成功，候选 `ab454866942b38e5f01b55ad91fa792bda03cc5d4bd70e40a3174bf18da7ce82`，`reused=true`，files/tables明确列出新表和兼容表。
- 正式性能表SHA256=`953592e594cdccfa7aed8914f5fed791cc10a0e972e6bc585ee4b5c3be751699`，Codex逐字节对照用户已批准样表完全相同；因此此前非法输入/吞吐、模板计数和布局验证仍适用于相同文件，无需重复整套样表验收。
- 兼容表SHA256=`2dabd3af336e08b59cc847b7cc8d5c89fc8bc46244abf3a685a0077cc365b05e`，保持既有字节。
- 新导出器及实际依赖进入DeepSeek tool_sources与analysis_id；run_manifest记录新表路径和交付哈希。产物经统一materialize/cache完整性路径处理，不存在静默遗漏新表的成功候选路径。
- Codex独立运行 `test_hf_build_performance_table.py`：12/12通过，无跳过。包含正式表/旧表、身份/清单、缺失/损坏/健康复用、缺依赖具名失败、单层和其他模型隔离、archify路径测试桩等。
- 主工作区无需再应用隔离patch；CLI/候选README/根README已给出正常生成和表格入口。

## 交付与边界轴：通过

- 两个复核代理分别只读检查需求与交付，不互相代签。独立run_hf_build得到相同候选与新/旧表哈希，真实数据与样表输入一致。
- 接手基线91份只读源码/输入保持；157历史候选的785个键路径保持；E01 workspace及批准/旧证据聚合哈希保持。
- 已批准4个Excel模块/脚本/测试的主源码与E01源文件哈希相同。接手前4个源码/文档备份与baseline一致，当前差异只涉及授权的build/CLI/两README及5个新增Excel/集成测试文件。
- 6张渲染与其重算副本来源绑定通过；正式表与批准样表字节相同，没有版面新变化，不重复大规模图像目检。
- Codex执行相关文件 `git diff --check` 通过。

## 结论与限制

E03主工作区接入通过，无剩余工程阻塞。用户已批准布局，E01/E02/E03本轮目标完成；GitHub #22/#23仍保持开放，遵守本轮不自动关票、不提交推送、不恢复自动化的边界。

原生Microsoft Excel未实际验证。真实Archify全图库未重建，本次只做其入口的测试桩集成检查；既有HTML源与历史产物保护已核，不能将此表述为新候选全图库浏览器验收通过。

正式命令：

```bash
python structure_reader/scripts/read_model_structure.py build-hf \
  --model deepseek_v4_flash_0731_hf --all-variants
```

主性能表位于上述候选的 `tables/decoder_performance.xlsx`。单层旧入口与其他模型本次未推广新布局。
