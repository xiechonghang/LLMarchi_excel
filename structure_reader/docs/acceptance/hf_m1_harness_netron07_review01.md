# 反馈07：Netron 链接交付复核（2026-10-21）

当前候选：`4a8fc8f1744405df41cbf3f45961285f9708c3a99eada112e47580138d8417be`。
入口：[新版 HTML](../../outputs/deepseek_v4_flash_0731_hf/4a8fc8f1744405df41cbf3f45961285f9708c3a99eada112e47580138d8417be/viewer/decoder.html)。原 DeepSeek Harness App / DeepSeek-V4.1-Flash 1017 High 实施，Codex 派单与复核；未直接开发产品。

结果：取消右侧并排 ONNX SVG/详情面板，保留原详图；36 模块结果在图下提供“在 Netron 网页打开”入口，默认实际简化文件，CSA/Indexer 和 Prefill/Decode 分别对应。在线第三方环境明确提示，保留原始/简化/映射下载。没有本地服务、系统安装/协议或公共文件托管。

## 工程核验

Codex `codex_harness_netron07_review01/check.py` 实际 rc0：36 链接独立解析 HTTPS 域名、identifier、data fragment，解码后与各自 model.sim.onnx 字节完全一致；错模块和改坏字节两有限反例被拒绝。108 下载链接存在，页面 ID 唯一/内链闭合，没有并排容器。原36 SVG 与 032ea 直接逐字节有序比较一致；72模型及全部144 ONNX目录文件与788ee字节一致，JSON/config/xlsx字节相同；1723旧交付无漂移。

151交付SHA、18当前工具源和7输入当前复算一致；原始两次CLI argv数组/cwd/env/stdout/stderr/returncode已读，均rc0，同新ID，第一次新建、第二次reused。仅build.py新增链接数据、decoder_page.py展示与资产测试发生本轮改变；builders/modeling/原diagram等保持。旧187冻结只有这3个授权产品/测试修改及Codex授权需求/计划/progress更新，单列而不篡改旧freeze。

Codex仅复跑7资产检查通过、无skip；字节相同72模型/数学源复用前轮115受影响和额外5不等轴小例证据，不重跑旧数值套/Excel/模型门禁。Harness报告声称19项全过但所引tests.*日志未保存，本轮不以该声称作为原始测试证据，采用Codex本次保存的7项实际结果；不为日志措辞再派产品修复。

## 真实 Netron 样例

使用原in-app browser，在官方Netron页面按候选HTML实际提取的decode_moe_merge.sim.onnx URL直接导航（没有读取file URL、没有启动服务/换浏览器/CDP）。自动显示routed_view/shared_out → Add → mlp_output，均B×T×H；点击Model Properties确认ONNX v10、ai.onnx v18、phase=decode、module=moe_merge，cua截图已核。该样例的真实自动载入与属性交互通过。

覆盖如实：36链接全部做字节对应；只实际打开1个代表模型，未把该样例说成36全浏览器验证。未验证本地HTML实际点击/滚动/完整CSS，先前file限制不绕过；`html_browser_visual=blocked/not_verified`，`netron_browser_sample=passed`。网页版需网络，链接把本无权重模型字节交给在线Netron页面，不承诺离线或绝不传输。无需重新选模型文件的行为已在样例核实。

## 状态与证据

`display_static_engineering=passed`；`native_excel=not_verified`；HF真实数值等价未做；`user_reading=pending`。FP32无真实权重阅读特化、单专家模板、默认mask子集限制保持。M1整体阅读未由用户确认，不启M2/M3，自动检查PAUSED；未Git提交或恢复用户删除。

证据：`codex_harness_netron07_review01/{check.py,result.json,asset_tests.stderr,browser_sample.json,freeze.json}`；`codex_harness_netron07_dispatch01/{before.json,dispatch.json,protected_candidates.json}`；Harness `dpsk_harness_netron07_result01.md` 与 `logs/dpsk_harness_netron07_01_build{1,2}.*`；公开部署脚本证据 `netron07_evidence/`。旧反馈06及所有历史证据只读保留。
