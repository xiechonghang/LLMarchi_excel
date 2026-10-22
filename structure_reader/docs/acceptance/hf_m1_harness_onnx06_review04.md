# 反馈06最终工程复核04：实际ONNX/onnxsim与原图并排

状态：display_static_engineering=passed；onnx_synthetic_validation=passed（有限小图）；browser_visual=blocked/not_verified；native_excel=not_verified；HF_numeric_equivalence=not_verified；user_reading=pending。M1未整体确认，M2/M3不启动。

## 新阅读入口

[decoder.html](../../outputs/deepseek_v4_flash_0731_hf/788ee019821817dff2504938f5b51e02b59a2e714a957eff67dc4c3df9ccb975/viewer/decoder.html)

由同一DeepSeek Harness App / DeepSeek-V4.1-Flash 1017 High完成，Codex仅需求映射/派单/复核，未接手产品开发。工具按用户明确批准仅隔离装在structure_reader/.work/tools/onnx06；无torch/模型权重下载或HF导入执行。

每阶段18模块（CSA复合图对应CSA与Indexer两份模型），共36份原始ModelProto与36份onnxsim结果；HTML原36SVG在左/新36阅读图在右。右图来自实际sim ModelProto，采用可展开的阅读分组，分组不是sim融合；完整tensor名shape/op_type及来源可查，原/简模型与mapping/state可下载。总览保持导航聚合，不伪称计算模型。

FP32阅读特化、无真实权重、非HF数值等价已在页面简短说明；MoE单专家模板、不展开跨专家循环，mask只原默认因果/滑窗且不含2D padding。参数前置、原源码/双向定位、Attention⊃CSA⊃Indexer、缓存载体和配置选择均保留。

## 实际复核结果

- 当前152文件字节全部与Codex纯构建吻合（不落盘），候选同788ee019821817dff2504938f5b51e02b59a2e714a957eff67dc4c3df9ccb975；151项manifest交付SHA逐项一致，18工具源/7输入实际SHA一致。原始两CLI构建argv/cwd/env/stdout/stderr/rc齐全，rc0，同ID，第一次reused=false、第二次true。
- 72原/简ONNX全是实际文件，无重名node；原生与优化器新生节点均能查对应映射。8个onnxsim插入Reshape按实际producer/consumer邻接唯一关联至c_rope/i_pool_rope，明确派生身份而非源码原生节点；Codex从实际Proto独立复核邻接与locator。
- 36次onnxsim返回成功，check_n=0不做数值自检；两阶段各866→826节点，实际节点数与state逐项一致。不能把阅读分组后的框数称为优化后节点数。
- 144文件链接按viewer真实目录0断；HTML id无重复、内部目标0缺。原032ea的36SVG逐字节完整保留，JSON/effective_config/xlsx逐字节不变。1419旧交付文件逐项0漂移。
- 计算修复后的受影响115项（0skip）及额外B2/T3/Nh2/Dh6/G4/Ro2/Rk2不相等轴5项均已由Codex复跑通过。当前72原/简模型与3a7f逐字节同，builders.py/modeling.py字节同，故承接该有限数值证据，Codex本轮仅复跑7项资产/关联/页面检查全过（0skip），未重复全套。
- 已实际回放的计算错误：窗口0/1/2、缓存空/历史、mask窗口offset、CSA block_bias来源与哨兵、sink广播、逆RoPE轴、GroupedLinear分组、MoE选择/形状/共享边界，按生产builder与独立numpy或明确期望检查。它们不是HF真实权重等价证明，也不独立证明全部75算子每条数学语义。

## 覆盖与历史校正

- fix02报告所谓tiny配置T/Nh/G“互不巧合”不采信（实际均3）；Codex额外不等轴检查补足。
- fix02“72模型0缺映射”当时过强；新生Reshape共8个曾缺关联，旧报告只读，fix03与当前独立检查明确补足。72文件=36原模型+36简模型，不能称72个简化模型。
- fix03实际日志运行19项（资产7+其余数值类12），不采信其“未重复数值”描述；Codex本轮实际仅运行资产7，已有115+5数值证据以相同模型字节复用。
- review02的left_svg_equal=false是Codex过滤漏掉phase-onnx前缀，已用完整36原SVG字节匹配校正，不是产品漂移。

## 证据与停止点

Codex：.work/coordination/hf_m1/codex_harness_onnx06_review04/{final_check.py,result.json,identity_and_simplifier.json,asset_tests.stderr,freeze.json}；有限数值回放：codex_harness_onnx06_review03/{replay.py,result.json,tests.stderr,unequal_axes.stderr}。Harness：dpsk_harness_onnx06_fix_result03.md与logs/dpsk_harness_onnx06_fix03_*，历史初版/修复报告及候选保持只读。

Harness已正常完成空闲，等待实际阅读反馈；原OpenCode角色不恢复/不派旧单，不新增布局或全量门禁。自动检查保持PAUSED，用户删除两文件保持删除，未Git提交。浏览器真实字体/CSS/点击/滚动及原生Excel未核，不以静态SVG/Node桩或小合成ONNX代签。只新用户反馈/实际新错误再继续同App1017。
