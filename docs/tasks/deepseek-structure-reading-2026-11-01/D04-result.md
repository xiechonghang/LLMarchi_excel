# D04 结果：MoE 关系、Hash/TopK 与专家内部

日期：2026-11-03（Asia/Shanghai）。只实施 D04/#20；未启动 D05，未关票、
未 commit/push、未恢复/新建自动化、未访问旧 T14；E01 隔离。

## 1. 新候选

| 项 | 值 |
| --- | --- |
| 候选 | `structure_reader/outputs/deepseek_v4_flash_0731_hf/f5b8f0da862daf292a2eb342906218d9204124d9f40ab4cba3663e85e4319fe4` |
| 正式入口 | `viewer/structure.html`（2,276,433 bytes，sha256 `40f00919ead17351033a60298aba555361e5bf31897f5b59271eacb51481bd43`） |
| 图 | 67 张 Archify 图、71 个嵌入图位 |
| 证据 | `D04-verify/` |

构建命令：

```bash
PYTHONPATH=structure_reader/src \
  /Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -c "from structure_reader.cli import main; import sys; sys.exit(main(['build-hf','--model','deepseek_v4_flash_0731_hf','--all-variants','--archify-layers','--json']))"
```

## 2. 第三层 MoE 关系

`moe-<gid>` 改为 7 节点模块关系：

- MoE 输入 → Hash/TopK 路由 → dispatch/命中专家；
- MoE 输入直接 → 共享专家（与路由并行，不画 gate→shared 数据依赖）；
- 路由 indices 与 weights 分别进入 dispatch / 专家计权；
- 单激活专家模板（未展开）→ routed 加权回填；
- 共享专家 → shared 全 token；
- routed + shared → `mlp_output`。

关系层节点保留“未展开”说明，详细路由/专家/共享/汇合在第四层页签。

## 3. 第四层真实通路

每层组新增：

- `moe-route-*`：Hash 走 `tid2eid[input_ids]` 查表（input_ids/表来源），无 TopK 选择；
  TopK 走 scores + `e_score_correction_bias` 选择，纠偏只参与选择、不参与权重；
  原始 scores gather 计权、归一化/routed_scaling_factor 由真实 op 事实呈现。
- `moe-expert-*`：专家 slice → gate_up_proj[e] → clamp/激活 → down_proj[e] →
  按 token_idx/top_k_pos 取 topk_weights 相乘；模板一份，仅实际命中专家循环。
- `moe-shared-*`：独立 gate/up/down，全 token 输入；不消费 expert_tokens。
- `moe-combine-*`：routed index_add/View + shared 加和 → mlp_output。

Hash/TopK、单专家、共享专家均用真实 op inputs/outputs 推导边界；未知 id 具名报错，
只有真实 `[]` 才是标量；内部与图外双消费都保留对应边。

## 4. 参数与事实

- 关系/路由/专家/共享/汇合图只带本图必要参数：
  `E/Eh`、`K=num_experts_per_tok`、`Es`、`routed_scaling_factor`、`swiglu_limit`、
  `H`、`B/seqlen` 等；来源来自 config/modeling。
- 未虚构 Engram/latentMoE；未改 Excel 公式/计费/汇总。
- 旧 D01–D03 的两阶段、四层导航、整网/Decoder/mHC、Attention/QKV-O/缓存保留。

## 5. 验证

- **563/563 测试通过**（17 文件；MoE 关系/Hash/TopK/专家/共享/汇合回归）。
- **67/67** 图 native validate/deliver/check；67/67 重建 HTML 与交付字节一致。
- 67 图标签重叠 0、SVG 可见文字重叠 0、省略号 0；71 图位短 iframe 0。
- MoE 事实审计：4 组关系均 `route→expert`、无 `route→shared`、shared 从 input；
  Hash `tid2eid`；TopK `router_bias`；expert gate/up/down slice 均在；
  shared 无 expert_tokens；combine 输出 mlp_output。
- 专家 token 数保持符号 `num_expert_tokens`，不假设全 token/全专家。

## 6. 浏览器/视觉

本机 Chrome 仍因 sandbox/GPU 初始化失败；最终候选页面探针一次尝试 exit 1、0 截图。
此前 D03 的 644 项显示门禁属于旧候选；D04 新 71 图位需在授权 Chrome 按
`browser-runbook.md` 复跑，不能沿用旧截图。用户按 `review-guide.md` 看 MoE 关系、
Hash/TopK、单专家、共享专家、汇合；视觉项标 **待用户审查**。

## 7. 保护

- 27208 输出文件 **27208/27208 未变**，硬变更 0；7303/8136 旧候选只读；
- 22242：22240/22242，2 个动态例外 `.DS_Store`、`monitor-state.json`，未回滚；
- 协调 5144：5143/5144，`tickets.json` 为外部派发更新；
- D01/D02/D03 接受候选、两处删除保持；Excel 最新两候选逐字节一致；
- E01 隔离树未触碰；当前 dirty 126（外部 `E01-verify/`、`dpskperfexcel草稿.xlsx` 未纳入）；
- 源文件 222：217 未变、5 修改（build_hf_decoder、archify_layers、hf_reading_page、两个测试）。

## 8. 停止点

未做 D05；未关 Issue、未 commit/push、未恢复自动化；停待 Codex 技术复核与用户视觉
审查。新增图与旧表粒度差异已按图注/详情登记，Excel 不变。
