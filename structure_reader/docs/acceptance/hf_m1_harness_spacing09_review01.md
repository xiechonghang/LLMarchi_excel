# 反馈09：Decoder 总览展开复核（2026-10-21）

候选 `2877a154d43fc0dcf8ec34431d5d42ac54b8d440301903c41e182ef0616edc75`。
[新版HTML](../../outputs/deepseek_v4_flash_0731_hf/2877a154d43fc0dcf8ec34431d5d42ac54b8d440301903c41e182ef0616edc75/viewer/decoder.html)。原DeepSeek Harness App / DeepSeek-V4.1-Flash 1017 High完成，Codex派单与复核，未直接开发产品。

按用户实际拥挤截图，只展开Prefill/Decode两张Decoder总览：增加列间空白96、行走廊空白52、容器表头留白34。graph自带spacing仅总览使用，全局默认不变；没有缩小字体、隐藏依赖或换层级。实际两阶段Q/KV/CSA横向框间距181→277，Q层到mask层77→129、mask到eager核58→110；画布1930.31×1423→2436.23×2167，保持原生尺寸滚动。

## 实际检查

Codex `codex_harness_spacing09_review01/check.py` rc0。与派单保存的旧源码构造结果独立比较：每阶段30节点、44聚合边、3层容器的id/kind/名称/说明/来源链接/边src-dst-kind-label-cond-title/接口关联和包含children完全一致；这里只证明总览原44语义边保留，不扩大成模块432方向重验。实际SVG全部text/tspan/title的文字多重集相同（几何改变导致绘制顺序变动正常），36SVG XML合法；仅下标0/18两张总览字节改变，34模块详图逐字节不变。

Codex复跑7针对性检查通过无skip：真实包含并列、cache载体和成对接口、具名main↔cache避core及q→out避mask、dense接口、页面XML/内链、本地Netron36链接真实resolve与108下载。Harness图示54项原始stderr/rc0已读，不重复全套。两实际CLI原始argv/cwd/env/stdout/stderr/rc已核，同新ID/第二次reused；151交付/21工具源/7输入当前SHA吻合。2179旧交付保护未变。

72原简ONNX、144ONNX目录文件、JSON/config/xlsx与aa581逐字节一致；此前032ea没有ONNX，不能把模型文件旧同表述成与032ea比较。启动器3源码和实际binary/Info与反馈08冻结一致，没有重编译/注册/触发协议，当前36链接仅随新candidate正确更新。数学/模型/Excel证据复用，不重跑旧115/19数值套或旧门禁。产品源码diff只有diagram.py的小范围总览spacing取值；其他派单保护文件不变。

## 可见性与边界

Codex用已安装系统QuickLook把实际SVG加当前页面原CSS作静态缩略图，已看展开后的间距和总体布局；该诊断不运行浏览器/Netron，不涉及被拒绝的协议，也不证明浏览器CSS/主题或用户实际体验。没有安装下载工具。静态图渲染器对CSS主题继承可能与浏览器不同，本轮不据缩略图调整原配色。

Harness的“边标签压框0/表头压节点0”是近似诊断，label高度固定12且拼接多行，node_frame_overlaps检查整容器而非表头，不能作为完整表头/标签碰撞独立证明；不采信其过强覆盖。当前真实间距增加、文字/依赖保留和具名路线回归由Codex上述检查补足。旧历史路由并未按数百交叉启发式扩大本轮；实际阅读还需用户看新版。

`display_static_engineering=passed`（本次总览间距范围），`browser_visual=not_verified`、`user_reading=pending`；反馈08真实本地协议点击仍blocked/user_action_required，未尝试任何绕过。M1整体阅读未确认/M2M3未启、自动PAUSED、Harness正常完成空闲；未Git提交/恢复用户删除。

证据：本目录check.py/result.json/targeted_tests.stderr/freeze.json及after_static.svg.png（静态缩略图）；派单codex_harness_spacing09_dispatch01/含用户截图、旧源码/两SVG/overview_semantics_before.json/2179保护；Harness dpsk_harness_spacing09_result01.md、spacing09/及logs/dpsk_harness_spacing09_01_*。历史冻结不改。
