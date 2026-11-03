# DeepSeek 正式集成规格：代码依据与现状核对

日期：2026-11-01。仅只读核对当前实现，不宣称运行测试或正式验收。代码位置是本次调查定位，后续重构可变化。

## 已有能力与实际差距

- 公开构建接口：`structure_reader/src/structure_reader/build_hf_decoder.py:1134` 的 `run_hf_build` 已提供全变体与 Archify 选项；单层与分层模式非法组合具名拒绝。可复用为规格主验收边界，不新建旁路构建器。
- 事实与语义接入：同文件 `:1261` 的 `_build_archify_layers` 已接收整网和层组结构；DeepSeek通过 `build_all_layer_group_bundles` 与 `normalize` 生成 raw/normalized。语义来源可复用；新增图的真实子图成员和参数仍需核验，不能照抄原型手写节点。
- 来源身份：同文件 `:1394` 左右维护 `archify_product_sources`；当前覆盖渲染、模型声明、语义规则/bundle和DeepSeek提取。后续若新增影响产物的代码，应纳入有效来源记录。未声称已验证所有传递依赖闭包。
- 页面：`structure_reader/src/structure_reader/render/hf_structure_page.py:58` 公开渲染已有整网/层组/Block/算子、离线源码快照及 Archify 参数；`:82` 组装页面。可以扩展同一入口，而非将原型页面直接改名发布。
- 双套绘图仍存在：同文件 `_network_section`（约`:199–225`）先 `render_svg`，再 `_archify_panel`，证明“只保留Archify”是待改需求，不是已实现行为。
- 场景：同文件 `:176` 与 `:741` 的 `_scenario_bar`、`_router_script` 仍有动态场景及数值派生面板。规格要求阅读UI两阶段、符号贯通；旧场景数据的移除需与Excel/其他消费者隔离，不能全局删掉。
- 图抽象：`structure_reader/src/structure_reader/render/archify_layers.py:327` 的 `build_network_graph` 已由整网真实operators/tensors投影，并表示层循环；可扩展端点与导航层级。
- 标签/布局：同文件约`:2095–2192`处理显式路线与标签位置，`:2244–2246`还有按宽度补省略号逻辑。说明全图完整文字、自动画布包容和标签关联需要正式实现；字段非空和节点坐标正常不能替代最终显示检查。
- Archify封装：`structure_reader/src/structure_reader/render/archify_preview.py:110`、`:186`、`:1349` 分别已有工具发现、工具身份和最终生成封装。沿用原生工具及回执，不改工具保护设置以获得通过。

## 测试先例（已阅读定位，未运行）

- `structure_reader/tests/test_hf_decoder_reading.py`：四层组、mHC comb方向、缓存顺序、Indexer、专家链、离线页面、完整标签、长参数换行、公开构建原子输出、Excel确定性等先例。
- `structure_reader/tests/test_deepseek_semantic.py`：四层组语义、具名投影/归一化分离、View/Transpose、全称轴、历史长度边界及状态消费者。
- `structure_reader/tests/test_archify_layers.py`：整网状态/辅助输入、全称shape、真实层组及Decoder成员关系，可用于新增正式分层图的回归定位。
- `structure_reader/tests/test_archify_layers_models.py`：共享构建器的多模型隔离等已有先例；本轮不开发其他模型，但影响共用模块时使用必要保护检查。
- 草稿边界记录 `.archify/architecture-deepseek-review-v8-20261101-214044/all-graphs-boundary-check.json` 覆盖21图路线点与节点/标签矩形；不能覆盖浏览器实际字体、动态隐藏、iframe或交互裁切，不能升级为正式验收证据。

## 状态边界

第八版有用户展示认可记录，正文与本规格明确浏览器目检未运行。T14协调文件仍记录第三轮待返修/待复核；其实现报告另有第三轮交付。当前GitHub核对目标为 `xiechonghang/LLMarchi_excel`，开放票为 #15（T14）；本次不协调旧状态、不关闭旧票。

本规格是新的DeepSeek阅读集成范围；代码可复用不代表旧交付满足规格。实际实现仍委托dsh，正式拆票及派发是后续步骤。
