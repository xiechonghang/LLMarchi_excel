# D04 rework 结果：IndexAdd、路由公式、MoE 身份与共享 Clamp

日期：2026-11-03（Asia/Shanghai）。只返修 D04/#20；未启动 D05，未关票、
未 commit/push、未恢复自动化、未操作 E01/旧 T14。误派期间已修改的两个主仓文件
与既有部分构建已核对并保留，未回滚。

## 1. 新候选

| 项 | 值 |
| --- | --- |
| 候选 | `structure_reader/outputs/deepseek_v4_flash_0731_hf/6ec5957a53a7ad82710eabf8d9bc0287b54789fe6499ef9319777102cfab5fc2` |
| 正式入口 | `viewer/structure.html`（2,285,219 bytes，sha256 `09a5d7988a4aee1e956ec3872fa34c0170154009b251fa7eb6a4da057a34fc39`） |
| 图 | 67 张 Archify 图、71 个嵌入图位 |
| 证据 | `D04-rework-verify/` |

构建命令：

```bash
PYTHONPATH=structure_reader/src \
  /Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -c "from structure_reader.cli import main; import sys; sys.exit(main(['build-hf','--model','deepseek_v4_flash_0731_hf','--all-variants','--archify-layers','--json']))"
```

## 2. 四项响应

### P1 真实 moe_combine/index_add

- `moe-combine-*` 加入真实 `moe_combine` 阶段 `c-add`：
  `expert_down_out + expert_token_idx → index_add_(0, token_idx, …) → routed_out`，
  再 `moe_routed_view`→View→`moe_merge`。
- 节点/边注明 `final=zeros_like` 起步、命中专家循环累加；`expert_down_out`
  保持 `num_expert_tokens,hidden_size`，`token_idx` 为 `num_expert_tokens`。
- 关系层 `r-expert→r-combine` 改传 `expert_down_out`，并补
  `r-dispatch→r-combine expert_token_idx`；不再把已全局回填的 `routed_out`
  冒充单专家输出。
- 专家权重只在 `moe_expert_weight` 乘一次；`c-add` 不消费 `topk_weights`。

### P2 路由公式主图可见

- `moe-route-*` 的 `m-router` 主副标题给出完整顺序：
  - TopK：`logits→score_fn=sqrtsoftplus(scores)→+e_score_correction_bias→topk indices→
    scores.gather(原始 scores)→/(sum+1e-20)→×Scl`；
  - Hash：`logits→score_fn=sqrtsoftplus(scores)→tid2eid[input_ids] indices→
    scores.gather(原始 scores)→/(sum+1e-20)→×Scl`。
- 纠偏只用于选择，不进入 gather 计权；未再放无连接的“公式节点”，公式在可达主节点上。
- 不机械微拆为伪张量，不造第二套事实。

### P3 MoE 真实身份 + 完整 shape

- `_edge_display_text` 对所有 `moe*` 图统一输出 `tensor_label｜shape`，不再只显示 shape。
- 实际 candidate 中可辨认：
  `indices / weights×Scl`、`token_idx`、`top_k_pos`、`expert_down_out`、
  `token_idx` 回填、`gate_up_proj[expert_idx=e]`、`down_proj[expert_idx=e]`、
  `shared gate/up` 等。
- 同 shape 的 `token_idx`/`top_k_pos` 由真实 tensor_label 区分；未用 tooltip/JSON 替代。

### P4 共享 Clamp/参数与 mHC 来源

- `moe-shared-*` 明确：
  `shared.gate_proj→clamp(max=Lim)`、`shared.up_proj→clamp(±Lim)`、
  `SiLU(gate)·up→down_proj`，全 token，不消费 `expert_tokens`。
- 共享参数卡保留 `Es/H/Lim`，去除无关 `hc_eps`；路由卡仅 `B/seqlen/H/E/K/Scl`；
  专家卡含 `E/K/Es/H/Lim`。
- 恢复 mHC 就近来源：
  `config：hc_mult、hc_sinkhorn_iters、hc_eps；modeling_deepseek_v4.py:926-938、976-979`，
  不再用“来源随参数卡给出”的空说明。

## 3. 验证

- **565/565 测试通过**（17 文件；含 IndexAdd、公式、身份 shape、共享 Clamp、mHC 来源）。
- **67/67** 图 native validate/deliver/check；67/67 重建 HTML 与候选字节一致。
- 67 图标签重叠 0、SVG 文字重叠 0、省略号 0；71 图位短 iframe 0。
- MoE 审计四组：route→expert、无 route→shared、shared 从 input、Hash lookup/TopK bias、
  expert gate/down slice、shared 全 token、combine merge 均通过。
- 旧 804 项显示门槛本次未复跑；新候选需 Codex 绑定复核。

## 4. 误派部分变更核对

- 误派期间修改的 `render/archify_layers.py`、`tests/test_deepseek_reading_page.py`
  与已有构建均保留，未回滚；本轮在其上补齐公式/根因并重新构建。
- 未操作 E01 会话/进程/输出；未把其文件纳入 D04。
- 最终候选 **不沿用** 部分构建回执；所有证据重新绑定 6ec59…/09a5…。

## 5. 浏览器/视觉

本机 Chrome 仍因 sandbox/GPU 初始化失败；最终候选页面探针一次尝试 exit 1、0 截图。
D04 新 71 图位/804 项需 Codex 按 `browser-runbook.md` 补跑；用户按
`review-guide.md` 看 IndexAdd、路由公式、身份 shape、共享 Clamp。视觉标
**待用户审查**，未宣称通过。

## 6. 保护

- corrected 基线 28870 输出 **28870/28870 未变**，硬变更 0；旧 7303/8136/f5b8 只读；
- 协调 5144：5143/5144，`tickets.json` 外部派发更新；
- 22242：22240/22242，2 个动态例外 `.DS_Store`、`monitor-state.json`，未回滚；
- D01–D03 接受候选、两处删除、Excel 最新两候选一致；
- E01 隔离树未触碰；当前 dirty 127（新增候选输出未跟踪），外部 `E01-verify/`、`dpskperfexcel草稿.xlsx` 未纳入；
- 源文件 222：219 未变、3 修改（build_hf_decoder、archify_layers、测试）。

## 7. 停止点

未做 D05；未关 Issue、未 commit/push、未恢复自动化；停待 Codex 技术复核与用户视觉审查。
