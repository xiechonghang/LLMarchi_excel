# D01 全图库连线标签布局返修结果

日期：2026-11-02（Asia/Shanghai）。范围：只做 D01/#17 的连线标签布局返修；
未启动 D02–D05，未关 Issue，未 commit/push，未恢复自动化，未操作旧 T14 会话。
状态：**代码、候选、原生校验和离线几何证据完成；浏览器标签验收待 Codex 补跑**。

## 1. 新候选

| 项 | 值 |
| --- | --- |
| 候选 | `structure_reader/outputs/deepseek_v4_flash_0731_hf/2fa51e64bd77da2b9328cd2548322311e37ead087833750185225ba1df32c97d` |
| analysis_id | `2fa51e64bd77da2b9328cd2548322311e37ead087833750185225ba1df32c97d` |
| 正式入口 | `viewer/structure.html`（2,019,910 bytes，sha256 `7edf05462803602bad406fa1653820f79a9690c03d8f40c6f74b3d7c84c80ea8`） |
| 整网原生图 | `viewer/archify/network.html`（sha256 `42c48b1a7d74c2e7a69f5e8385625e2334c252ef92d7570cd72894a06ccaf75d`） |
| 旧候选 | `f1ea240e…` 及更早候选全部保留；新构建写入新目录 |

证据目录：`D01-label-verify/`。

## 2. 根因与修复

Codex 复核在 1440×900/1920×1080 下测得 13 张图、每视口 66 对标签重叠，主要是：

- **平行边**：同一源/目标的多条依赖共用一个 `labelAt`（如 moe hash-lookup→router/
  dispatch、expert-clamp→act、head base/scale→pre）；
- **汇入边**：多条边在同一目标行上方使用同一 `labelAt`（如 q/mask→core、
  mhc/d-in→res、mhc-ffn/res-attn→res）；
- **跨层长边**：底部 d-out 的多条长标签全部落在同一行（每图 6 条），互相遮叠。

修复在共同生成器 `render/archify_layers.py` 中新增
`separate_edge_labels`，对全部图位统一执行：

1. 按每条边的真实路径（`labelAt` 或两端点 + `via`）计算可用 y 范围；
2. 同一行/同一目标的标签做 **11/14px 步长的垂直错层**，保持在边附近；
3. 垂直不行时，在路径 bbox 内做有限水平让位；
4. 仍不行才扩大上下偏移；底部输出边允许向下堆叠，保持靠近汇聚端点；
5. 节点矩形、节点标题文字（完整文字可能比节点框宽）都作为障碍；
6. 跨层长边的标签优先放在**目标行上方**的行间空隙，避免全部落到图底；
7. 不删标签、不改文字、不缩字、不隐藏；只调整 `labelAt`。

配套改动：

- `build_hf_decoder.py`：图位画布覆盖标签包围盒，嵌入 iframe 不再按组件尺寸
  裁掉标签；
- `hf_reading_page.py`：内嵌高度 padding 32→48px，覆盖 viewBox 底部；
- 标签移动后仍保留原 tensor id、来源、全称 shape 和边。

### 问题对应

| 图/模式 | 复核重叠对数（每视口） | 修复 |
| --- | ---: | --- |
| head（base/scale→pre） | 1 | 两条同文字标签错层 |
| decoder-swa-hash | 6 | q/mask→core、mhc/d-in→res、mhc-ffn/res-attn→res 等错层 |
| mhc-swa-hash | 1 | 长标签与“数据依赖”短标签错层 |
| moe-swa-hash | 5 | hash-lookup 两条平行边、expert-clamp 两条同目标、底部汇入错层 |
| decoder-csa-hash | 13 | 3 条汇入 + 6 条 d-out 长标签错层 |
| mhc-csa-hash | 1 | 同上 |
| moe-csa-hash | 5 | 同上 |
| decoder-hca-topk | 9 | q/mask/core、d-out 长标签错层 |
| mhc-hca-topk | 1 | 同上 |
| moe-hca-topk | 5 | 同上 |
| decoder-csa-topk | 13 | 同上 |
| mhc-csa-topk | 1 | 同上 |
| moe-csa-topk | 5 | 同上 |

所有 29 张唯一图 / 33 个图位都走同一套生成器；不是只改截图举例。

## 3. 测试

- 17 个测试文件、**540 项全部通过**（`D01-label-verify/tests.log`）。
- 新增 6 项标签布局回归：
  - 全部图组：任意两个连线标签不重叠；
  - 全部图组：标签不覆盖节点框；
  - 全部图组：标签不覆盖完整文字节点标题（含溢出部分）；
  - 平行边：moe hash-lookup→router/dispatch 标签错开；
  - 汇入边：decoder q/mask→core 标签至少 14px 错开；
  - 长标签：decoder d-out 底部 6 条标签不重叠且在界内。
- 既有阶段切换、符号化、页签隔离、端点权重/来源、四层导航测试保持通过。

## 4. 原生与页面证据

| 项 | 结果 |
| --- | --- |
| 29 图 `validate`/`deliver`/`check` | **29/29 通过** |
| 29 图重建 HTML 与交付字节 | **29/29 一致** |
| delivery sidecar 绑定候选 JSON/HTML | 29/29 |
| 页面契约 | 2 阶段；9 页签组/22 按钮；33 个 `?embed=1` iframe；路由 595、锚点 1901、相对文件 168，失效/缺失 0 |
| 嵌入图位 | 33；viewBox 高度不足 **0** |
| 29 图标签几何（离线估算） | 454 个显式 `labelAt` 标签 + 156 个 attention/kv 自动标签；显式标签重叠 **0**、SVG 可见文字重叠 **0**、省略号 **0** |
| 标签关联距离（离线） | 标签中心到所属边折线最大 **45px**，>80px 的标签 **0**（`label-distance-audit.json`） |

说明：454/156 和 0 重叠是**离线字符宽度估算**，不是浏览器 `getBBox`；
最终仍以 `verify-final.mjs` 的 424 项和截图为准。

## 5. 浏览器限制

dsh 当前环境仍无法启动 Chrome：

- `Failed to initialize sandbox. Operation not permitted`
- `GPU process exited unexpectedly: exit_code=6`
- `Timeout Page.enable`

已实际运行并保留原始回执：

- `D01-label-verify/probe-attempt/`（`--geometry-only`，exit 1，0 截图）
- `D01-label-verify/probe-attempt-full/`（完整探针，exit 1，0 截图）
- `D01-label-verify/visual-attempt/`（network visual-check，exit 1）
- `D01-label-verify/native-evidence.json`（29 图 browser-check 全失败；
  validate/deliver/check 全通过）

未关闭或绕过 Chrome sandbox，未改探针断言。因此：

- 未保存最终候选截图，未逐张目检；
- 未宣称 26 个失败图位/66 对重叠已在浏览器中消失；
- AC17 及浏览器运行时验收仍待 Codex 在授权环境补跑；
- 补跑命令见 `D01-label-verify/browser-runbook.md`。

## 6. 保护与边界

- 继承 22242 项历史产物：**22242/22242 未变**；
- 此前候选 248 个文件：**248/248 未变**；
- 派发前当前候选 `f1ea240e…` 的 124 个文件：**124/124 未变**；
- 两处既有删除保持；Excel 最新两候选逐字节一致（`2dabd3af…365b05e`）；
- 源文件快照 168 个：164 未变，4 个修改（archify_layers、build_hf_decoder、
  hf_reading_page、test_deepseek_reading_page）；
- `D01-final-codex-*`、`D01-display-verify/` 既有证据、协调 JSON、handoff 只读。

## 7. 停止点

停待 Codex 复核。未关 Issue、未 commit/push、未恢复自动化、未操作旧 T14 会话；
未提前实现 D02–D04 语义细分。
