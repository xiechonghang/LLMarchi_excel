# 反馈08：本地 Netron 链接集成复核（2026-10-21）

候选 `aa581b11774afea16e970dcf042f20a68f54cda1cce28e316bd0484d09bdf552`。
[新版HTML](../../outputs/deepseek_v4_flash_0731_hf/aa581b11774afea16e970dcf042f20a68f54cda1cce28e316bd0484d09bdf552/viewer/decoder.html)。由原DeepSeek Harness App / DeepSeek-V4.1-Flash 1017 High实施；Codex仅需求、派单、审查、注册与复核。

用户实际反馈网页链接不能一键打开，已装本地Netron。本次保留原详图，36模块入口改为“用本地 Netron 打开”专用本地链接，默认对应简化ONNX；CSA/Indexer各自入口、Prefill/Decode分别绑定。原网页版链接已退出当前页面，旧候选不改。

## 启动器和注册

已核 `/Applications/Netron.app` 9.3.2支持onnx文件但无URL scheme。项目内 `tools/netron_launcher/{main.swift,Info.plist,setup.sh}` 生成 `.work/tools/netron_launcher08/LLMarchiNetron.app`，注册专用 `llmarchi-netron` 链接。Swift仅把固定本项目候选/阶段/白名单模块的model.sim.onnx交给已有Netron，原生NSWorkspace文件打开，无shell/AppleScript/服务/网页上传；只读manifest交付SHA核验，拒绝未知/重复参数、userinfo/port、穿越、符号链接及非regular目标/manifest，SHA错误不会打开。

初版6310的无条件1.5s退出会打断冷启动，已由原Harness有限收尾：收到URL取消无URL计时，只在NSWorkspace完成回调后退出。没有把固定等待加长冒称可靠。无URL启动可短时自行退出。

Codex已审查源/Info/setup。setup在沙箱内编译成功，LS注册受沙箱限制失败-10822；随后仅对该项目app执行已授权窄范围lsregister -f，rc0。只读NSWorkspace查询（同样在允许系统LaunchServices查询的通道）实际返回本项目 `.work/tools/netron_launcher08/LLMarchiNetron.app`，不是仅凭注册命令成功宣称已绑定。未安装到Applications、未改Netron原包、未更改ONNX默认应用/浏览器安全设置，不建服务。撤销该专用关联可用已有setup.sh unregister。

## 受影响工程检查

Codex实际 `codex_harness_netron08_review02/check.py` rc0，8资产检查全过、无skip：36链接经真实编译启动器同一resolve-only代码解析到同候选对应sim，另独立核每条链接与所在h4模块/阶段绑定（Indexer对应CSA父图），108下载链接存在、内链闭合/ID唯一。有限错参数、穿越、manifest SHA不符/出根符号链接被拒绝，fixture仅新临时目录，未改历史。

原36SVG与032ea有序逐字节同，72ONNX与上版4a8f同，全部144 ONNX目录文件及三数据JSON/config/xlsx也逐字节同；2027旧交付无漂移。151交付SHA、21实际工具源（含3启动器源）/7输入与当前同源；新两次真实CLI原始argv/cwd/env/stdout/stderr/rc已核，同新ID、第二次reused。编译后实际binary SHA `859728cbcdf9102837e2e235e106f4336776e68361201fea40390fcfb6b07054` 与收尾一致，Info SHA `b75b39f2942c353b8a4802401679ebb5dbb653805d6b9457999bd28a02bd7f28`。无数学/模型/Excel变化，复用旧115+5数值小例，不重跑旧整套。初版称已存argv文件但未见，该口径不采信；本次新runner进程内原始argv已实际落盘，不伪补旧记录。

## 真实点击仍需用户完成

Codex通过原生cua启动了本地Netron并看到Open Model；这只证明App能启动。系统Spotlight UI通道超时，没有确认触发本地链接。随后使用原IAB尝试本次明确要求的专用链接，Browser Use安全策略明确拒绝非http/https协议，禁止替代通道/间接执行绕过。此后未另换浏览器/终端/open命令/协议/服务尝试达到同一被拒绝动作。

因此 `static_integration=passed`、`project_handler_registration=verified`，而 `native_link_end_to_end=blocked/user_action_required`、`html_browser_click=blocked/user_action_required`。不能把resolve-only或系统注册当作用户一键点击已通过。请用户在新版HTML点击“用本地Netron打开”，如浏览器首次询问打开项目启动器按系统提示确认；需用户反馈实际对应图是否显示。Codex不代签。原浏览器CSS/原生Excel/用户整体阅读分别仍未核，M1未最终确认，M2/M3未启、自动PAUSED；未Git提交/恢复用户删除。

证据：本目录check.py/result.json/asset_tests.stderr/registration_verified.json/ui_verification.json/freeze.json；初始派单codex_harness_netron08_dispatch01/；有限收尾保护codex_harness_netron08_review01/；原Harness dpsk_harness_netron08_fix_result01.md、logs/dpsk_harness_netron08_fix01_build{1,2}/argv.json及同前缀stdout/stderr/returncode。历史只读保留。
