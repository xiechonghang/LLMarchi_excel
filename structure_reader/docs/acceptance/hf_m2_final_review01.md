# M2四组合最终静态工程验收

2026-10-24。用户已认可M1阅读并授权分析其他DeepSeek V4实际结构。本轮由原DeepSeek Harness App会话、DeepSeek-V4.1-Flash 1017 High实施，Codex派单/需求映射/验收，未自行开发产品。第20至40轮的中间报告、失败与修复记录保留，未恢复旧OpenCode角色。自动检查保持PAUSED，未Git提交或推送，用户删除文件保持删除。

最终身份：`570b4a27e3f6de0bfb255a33a9cf1899257b72d63b153d7d3bf0af8ef763f551`。

阅读入口：[四组合HTML](../../outputs/deepseek_v4_flash_0731_hf/570b4a27e3f6de0bfb255a33a9cf1899257b72d63b153d7d3bf0af8ef763f551/viewer/decoder_variants.html)，[两表Excel](../../outputs/deepseek_v4_flash_0731_hf/570b4a27e3f6de0bfb255a33a9cf1899257b72d63b153d7d3bf0af8ef763f551/tables/decoder_variants.xlsx)。候选共562文件，run_manifest自体不列其自身hash，清单561交付。

## 用户可见成果与需求对应

- 依当前根config及固定HF配置转换规则，43层恰好分为四组（层号从0起）：SWA+Hash层0/1、CSA+Hash层2、HCA+TopK层3/5/…/41、CSA+TopK层4/6/…/42；各用代表层模板，保真实归属，不复制43份相同结构。
- HCA展开非重叠128窗口、独立kv/gate投影、闭窗聚合/RMSNorm、条目位置RoPE、历史追加与余量/entry_count回写；没有CSA overlap和Indexer。SWA展开普通main RoPE、无压缩器/Indexer、DynamicSlidingWindowLayer独立keys/values字段及当前可读full_key_states；注意力实际以update返回[0]作共享K/V。Hash查tid2eid选择专家，学习分数只对所选专家gather/归一/缩放，保专家dispatch与共享专家通路，不误作TopK选路。
- HTML先完整Prefill四组，再完整Decode四组；保纵向算子图、完整参数/维度名、输入到输出讲解、真实缓存载体及读写、源码展开和算子回图。CSA中Indexer仍为内部结构。新变体沿用已核图样式，未另加总体布局目标。
- Excel仍仅Prefill/Decode两表，各按四组列参数与完整14列算子区；配置值/来源/本表本组公式保留，I/O/权重维度逐轴使用完整名。
- 138模块实例各有原始与onnxsim简化ONNX，共276模型；SWA每阶段16模块、HCA17、两个CSA组各18。页面链接绑定本候选/本组/本阶段，原本地Netron启动器仅增加白名单group参数，旧四参数链接兼容。

## 实际复核

Codex新有限诊断位于`.work/coordination/hf_m1/codex_harness_m2_final_review01/`：

- `check_delivery.py`与`delivery_result.json`：142 SVG解析，554计算框源码名/类型全文，1822条可见边标签全文；1538条详情实际路径起终点定位到真实circle/IO/字段时态端口/计算框，再核JSON生产/消费引用。字段文字溢出0；内部href/源码开关/回图目标/marker闭合、跨组唯一id检查通过。导航总览未计入1538的JSON方向证明。
- 实际xlsx回读：两表四组、554算子行、266参数行，全部I/O/权重逐轴shape、参数含义/数值/来源、本组公式和参数超链核对通过。此前逐字符shape、HCA计算类型截断、SWA详情旧别名和16处字段高度遗漏已关闭。
- `check_onnx_links.py`与`onnx_links_result.json`：276模型实际checker、输出state/简化返回true/模型SHA/组阶段及代表映射核实；138新链接经实际已编译binary `--resolve-only`成功，旧7b链接仍成功。启动器SHA `b7b10037e451e45f56aace3cfad3982d12589d11ffcde2dbb0d3e068efbf94e1`，未更改系统注册或权限。
- HCA生产3个有限检查、SWA/Hash3个有限数值检查、mask/Q3个有限检查曾由Codex复跑通过；采用同生产Ctx/GB和独立NumPy期望，覆盖多窗口非零entry_count/batch2/RoPE交错、零闭窗、独立滑窗字段/sink/AV、Hash查表和dispatch。原简与独立期望的具体覆盖以实际测试为准：mask小例部分断言只比阻断位或原始输出，不据此宣称全部原简数值等价。全模型HF数值等价未主张。
- 最后`test_m2_guard.py`两个首组SWA内存失败例由Codex复跑通过；检查已移入四组循环，checker/shape/simplify失败不能仅因最后CSA组正确而发布成功。第39轮此处循环位置遗漏保留在历史报告，不移植其通过结论。
- `build_identity_result.json`与实际保存runner/原始日志：第40轮两个真实子进程CLI构建rc0、stderr空，同最终ID，首轮reused=false、次轮true。真实argv/cwd/解释器/必要依赖环境均可回放；第38/39轮手设sys.argv的旧driver记录仅属库入口参数记录，不冒称本次真实系统argv证据。
- 28真实工具源（含所有ONNX新helper/Swift）及7输入当前SHA与manifest一致，身份可复算。冻结单层v1契约SHA仍`fff5c23b8e21a4980390191011011e9a4a3468552aea33a742cf6c02d4cff6dd`；默认层4数据与已认可7b逐对象同。相较a2ec，最终556个ONNX目录文件（含四报告）、JSON、xlsx逐字节同，HTML仅候选ID变化。10289历史候选文件保护复算无漂移。

最终冻结清单`final_freeze.json`记录本次实际交付、源、输入、必要小测试与本报告；历史冻结/候选/独立checker均只读。本轮新检查的别名helper作为被测对象核显示完整性，源码语义核查限新增/具名及代表，不将554/164等存在计数称为逐个独立证明。

## 结论与边界

`m2_static_engineering=passed`；`browser_visual=blocked/not_verified`、`native_excel=not_verified`、`netron_protocol_gui=not_verified`、`user_m2_reading=pending`。静态SVG/库回读/resolve-only不等浏览器字体CSS、真实点击或原生Excel，未绕过原浏览器file/协议限制。ONNX为无真实权重的FP32阅读特化，默认因果/滑窗mask未覆盖padding，专家为单专家模板；shape inference成功仍可含动态未知维。当前仅实际Decoder配置变体，未展开整网embedding/hyperhead/logits/Skill，未采样。

M2开发与本轮静态工程交付完成，等待用户阅读反馈；M3尚未启动，不因等待而派旧单或新增任务。
