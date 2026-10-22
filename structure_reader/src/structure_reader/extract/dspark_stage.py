"""DeepSeek-V4-Flash-0731 DSpark 公共 stage 语义图与三实例绑定（工作包 B · 步骤 B2）。

数据来源：models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/{model.py,kernel.py}
与 inference/config.json（固定 SHA256 基线，:mod:`structure_reader.extract.evidence`
逐条核对原文）。静态取证，不导入/不执行模型代码。

B2 范围（整体 B 仍 in_progress，本模块不是完整 DSpark 交付）：
1. 公共 stage SemanticGraph（decode 完整链 / prefill 仅 cache 构建）：
   真实 h 输入消费边（stage.x_hc_in_src → 首个 hc_pre_attn.flatten，
   decode 相位；prefill 走 Block 级 passthrough，DSparkBlock.forward
   845-849）；两次 hc_pre→norm→真实 DSparkAttention/TopK MoE→hc_post，
   归一化因子（model.py:683-684 rsqrt(mean(x²)+eps) 乘 F.linear 输出）
   显式成链；hc_fn/scale/base、attn_norm/ffn_norm、gate weight/bias
   参数节点与真实消费边（参数身份静态存在，到 decode-only 算子的
   消费边挂 g_decode）；residual/post/comb 因果边完整。全部 decode
   专用骨架运行时消费边挂可求值 g_decode（prefill 逐边求值屏蔽，
   DSparkBlock.forward 845-849 只建 cache 的早退）；main_x→共享
   mainKV 两相位不挂 guard。Attention 复用 B1 已落盘子图：内部节点/
   边与 B1 单一实现同源（_add_attn_flow 单份连线），fold_group 成员
   = 当前 stage 实际创建的节点集合（逐成员在当前图验证存在），边界
   映射（boundary mappings）显式映射外部端口到 Attention 内部节点
   端口，不是 opaque 名字丢内部连线；MoE 复用参考提取的 TopK 公共
   结构（gate score→bias→topk→gather→dispatch→专家模板→shared
   expert→accumulate，与 deepseek_v4_reference._add_moe 同构）+ 真实
   路由因果（dispatch 消费 indices 产出每专家 token 数 counts 与
   (idx,top) 路由选择对，经 select/control 边到专家模板——counts 决定
   空专家跳过、routing 决定 token 子集与权重槽位，model.py:639-645；
   N_e 数据相关静态未知，不编造数值），真实端口映射，不复制 CSA 压缩
   路径、不重写通用框架。
2. 三真实实例绑定与角色记录（可规范序列化往返）：stage_id 0/1/2、
   actual layer_id 43/44/45（DSparkBlock constructor 编号 = n_layers+mtp
   序号，model.py:902；主干仍 43 层，非 46 个 main layers）、mtp.0/1/2
   独立参数路径与 win_cache state_id；首/末布尔角色由 stage 数量纯逻辑
   计算（single stage → first 与 last 同时 true）。first 专有 main_proj/
   main_norm/forward_embed；last 专有 norm/hc_head 参数/markov_head/
   confidence_head/forward_head；共享 embed/head 引用 Transformer 同一
   对象（model.py:903-904），不复制三份权重。角色专有模块记录为后续
   子图 pending，不冒称已提取首末图。
3. 参数/状态展开绑定表（B2 补修 01 新增，task_b2 展开关系交付）：
   公共模板参数源/算子参数槽 → 各 mtp.i 实例路径（真实 shape/dtype/
   证据锚定，涵盖 Attention 参数、公共双 mHC hc_fn/scale/base、
   attn_norm/ffn_norm、TopK MoE gate/expert/shared expert 与 first/last
   角色专有参数；gate.weight 存储 dtype = 参考入口 torch 默认 dtype
   bf16——model.py:562 torch.empty 无显式 dtype + 950
   set_default_dtype(bfloat16)，见 ev_st_entry_default_dtype）；模板
   cache 张量/state_id/版本对/transition → 各实例独立身份。可规范 JSON
   往返，读回以固定可信来源核对（B2 补修 02）：期望参数目录 = 固定
   ModelValues 派生的完整规范目录（spec 逐字段一致——spec 与 bindings
   共同篡改 shape/dtype 不通过）、instance_path 恰为
   mtp.{stage_id}.{param_slot}（含源码模板通配槽，非 startswith 前缀）、
   n_stages/n_main_layers 与固定 config 一致、state 模板身份/版本对/
   transition/cache 张量 name 与真实 stage 图对应（构建期逐项核对防
   常量漂移）——实例间复制身份、整槽消失（spec+bindings 同删）、假
   state/张量名、以被检文档自身拼期望均具名拒绝。ChainMap 读回复用
   同一绑定一致性校验。
4. 最小三 stage 链路图/边界映射：同一 main_x 分发三 stage
   （forward_spec model.py:929-934）、draft h 串接 stage0→1→2、输出
   留待 head；prefill 三 cache 构建但草稿 h passthrough（DSparkBlock.
   forward 845-849；forward_spec for 循环在 prefill 仍执行并传递 h，
   各 stage prefill 分支 passthrough 返回原 h——不算草稿双 mHC 计算，
   chain 边显式携带 phase/guard 字段）。chain.embed_h 源是
   forward_embed 产出的 HC 展开流（非原始 transformer.embed.y 直接当
   HC 流）；readout 汇聚（40/41/42→mean(hc)→有序 concat→stage0
   main_proj/main_norm）记录为有证据的 binding/consumer，不把各
   readout 错误对应到各 stage。run config compress_ratios 46 项 =
   前 43 主干（消费层 0-42）+ 后 3 draft（消费层 43-45，全 0）；root
   config num_nextn_predict_layers=1（checkpoint nextn.* 单 MTP 段）、
   run config n_mtp_layers=3（DSparkBlock 实例数）分别记录来源/消费者。
   固定输入不修改。

不做（后续步骤续接）：first 入口（embed→HC 展开→hc_pre_attn 完整入口
内部图）与 last 出口（hc_head→norm→head→Markov 采样循环→Confidence
完整出口内部图）的完整子图提取、Markov/Confidence 头内部图、全量
集成（build/render/export/contracts/layer_plan）。本模块只交付公共
stage 模板 + 三实例绑定 + 链路边界映射，全部边界与 pending 显式记录。

可调用入口：
- build_dspark_stage_template()  → DSparkStageBundle（公共 stage 图 +
  证据 + 数值 + 坐标表达式记录，坐标语义同 B1 修正：坐标值非存储 view）
- derive_stage_bindings(n_stages, n_main_layers) → tuple[StageBinding,...]
  （纯角色逻辑，不修改固定配置）
- serialize_bindings / deserialize_bindings（公开 JSON 往返）
- build_chain_map() → ChainMap（三 stage 链路边界映射 + readout binding
  + compress_ratios 消费说明）
- chain_map_to_json / chain_map_from_json（公开 JSON 往返）

失败异常接口（以实际接口为准）：证据核对失败抛
structure_reader.extract.evidence.EvidenceError；图构造错误、
validate_graph_references 未通过、binding/chain JSON 结构不合法抛
structure_reader.extract.deepseek_v4_reference.ExtractionError
（ValueError 子类）。均具名拒绝、失败不给图/不给绑定。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.deepseek_v4_dspark import (
    ATTENTION_TEMPLATE_ID,
    EXPR_DRAFT_POS0,
    EXPR_SLOT_WRITE,
    TEMPLATE_ID as _B1_TEMPLATE_ID,
    CoordinateExpression,
    _add_attn_flow,
    _add_attn_tensors,
    _attn_coords,
    _axes as _attn_axes,
    _guards as _attn_guards,
    _states as _attn_states,
    build_dspark_attention_graph,
)
from structure_reader.extract.deepseek_v4_reference import (
    ExtractionError,
    _GraphBuilder,
    _evidence_defs as _reference_evidence_defs,
)
from structure_reader.extract.evidence import (
    EvidenceBase,
    EvidenceRegistry,
    SourceEvidence,
    load_evidence_base,
)
from structure_reader.graph_document import SemanticGraph
from structure_reader.graph_ir import FoldBoundaryMapping, FoldGroup, Guard, NamedAxis
from structure_reader.model_registry import find_workspace_root
from structure_reader.rules.deepseek_v4_flash_0731 import (
    ModelValues,
    load_model_values,
)
from structure_reader.scenario_evaluator import check_expression

__all__ = [
    "STAGE_TEMPLATE_ID",
    "BINDING_SCHEMA",
    "CHAIN_SCHEMA",
    "EXPR_MAIN_PROJ_IN",
    "TEMPLATE_STATE_ID",
    "TEMPLATE_STATE_TRANSITION",
    "TEMPLATE_STATE_VERSIONS",
    "TEMPLATE_CACHE_TENSOR_NAME",
    "StageBinding",
    "TemplateParamSpec",
    "InstanceParamBinding",
    "InstanceStateBinding",
    "StageExpansion",
    "ChainMap",
    "DSparkStageBundle",
    "build_dspark_stage_template",
    "derive_stage_bindings",
    "build_template_expansion",
    "serialize_expansion",
    "deserialize_expansion",
    "serialize_bindings",
    "deserialize_bindings",
    "build_chain_map",
    "chain_map_to_json",
    "chain_map_from_json",
]

STAGE_TEMPLATE_ID = "deepseek_v4_flash_0731:dspark_stage"
BINDING_SCHEMA = "dspark-stage-bindings/v1"
CHAIN_SCHEMA = "dspark-chain-map/v1"
EXPANSION_SCHEMA = "dspark-stage-expansion/v1"

# stage0 main_proj 输入宽（model.py:832）：args.dim × len(dspark_target_layer_ids)
# = H × N_tgt，三个主干 readout（层 40/41/42 各 mean(hc) 后 [B,T,H]）沿
# target_layer_ids 顺序 concat。
EXPR_MAIN_PROJ_IN = "H * N_tgt"


# ---------------------------------------------------------------------------
# 证据目录：本模块新增条目（stage/角色/链路）。hc/MoE/Attention 内部证据
# 复用 B1 模块与参考提取模块同源注册（同 ID 同原文，注册时逐条再核对）。
# ---------------------------------------------------------------------------

_E = SourceEvidence


def _stage_evidence_defs() -> tuple[SourceEvidence, ...]:
    m = "model.py"
    rc = "run_config.json"
    rootc = "root_config.json"
    return (
        _E("ev_st_hc_norm_factor", m, "Block.hc_pre rsqrt factor", 683, 685,
           "x = x.flatten(2).float()\n"
           "rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)\n"
           "mixes = F.linear(x, hc_fn) * rsqrt",
           "hc_pre 归一化因子：F.linear(x,hc_fn) 输出逐元素乘 "
           "rsqrt(mean(x²)+eps)（x 为展平后的 N_hcH 宽流）——mixes 不是纯 "
           "linear 输出，归一化因子链是 hc_pre 的真实因果步骤", "source_checked"),
        _E("ev_st_hc_params", m, "Block.__init__ hc params", 667, 678,
           "self.hc_mult = hc_mult = args.hc_mult\n"
           "self.hc_sinkhorn_iters = args.hc_sinkhorn_iters\n"
           "self.hc_eps = args.hc_eps\n"
           "mix_hc = (2 + hc_mult) * hc_mult\n"
           "hc_dim = hc_mult * args.dim\n"
           "with set_dtype(torch.float32):\n"
           "    self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))\n"
           "    self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))\n"
           "    self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))\n"
           "    self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))\n"
           "    self.hc_attn_scale = nn.Parameter(torch.empty(3))\n"
           "    self.hc_ffn_scale = nn.Parameter(torch.empty(3))",
           "公共双 mHC 参数（每实例独立）：hc_attn_fn/hc_ffn_fn "
           "[N_mix,N_hcH] fp32（N_mix=(2+N_hc)×N_hc=24、N_hcH=16384）、"
           "hc_attn_base/hc_ffn_base [N_mix] fp32、hc_attn_scale/"
           "hc_ffn_scale [3] fp32；set_dtype(torch.float32) 块内创建；"
           "DSparkBlock 继承 Block.__init__（822 super().__init__）——"
           "mtp.0/1/2 各有一套，非共享", "source_checked"),
        _E("ev_st_norm_params", m, "RMSNorm weight fp32", 189, 195,
           "class RMSNorm(nn.Module):\n"
           "    def __init__(self, dim: int, eps: float = 1e-6):\n"
           "        # rmsnorm in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.\n"
           "        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))",
           "attn_norm/ffn_norm（Block.__init__ 716-717 RMSNorm(args.dim)）"
           "参数 [H] fp32（checkpoint 存 bf16、模块内 fp32——源码注释原"
           "文）；main_norm/q_norm/kv_norm 同实现不同实例", "source_checked"),
        _E("ev_st_gate_params", m, "Gate.__init__ params", 551, 567,
           "class Gate(nn.Module):\n"
           "    self.topk = args.n_activated_experts\n"
           "    self.score_func = args.score_func\n"
           "    self.hash = layer_id < args.n_hash_layers\n"
           "    self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))\n"
           "    self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))",
           "TopK gate 参数（stage 层 43/44/45 ≥ n_hash_layers=3 → hash=False "
           "分支 bias 非 None）：gate.weight [E,H]（torch.empty 无显式 dtype → "
           "torch 默认 dtype；参考入口 model.py:950 set_default_dtype("
           "bfloat16) → 存储 bf16，见 ev_st_entry_default_dtype/"
           "ev_st_linear_dtype_mech；Gate.forward 570 weight.float()/x."
           "float() 是 fp32 求值，不是存储 dtype 证据）、gate.bias [E] fp32；"
           "score_func=sqrtsoftplus（run config:16）",
           "source_checked"),
        _E("ev_st_linear_dtype_mech", m, "Linear default dtype mechanism",
           129, 136,
           "class Linear(nn.Module):\n"
           "    def __init__(self, in_features: int, out_features: int, "
           "bias: bool = False, dtype = None):\n"
           "        dtype = dtype or default_dtype",
           "自定义 Linear 默认 dtype 机制：dtype = dtype or default_dtype——"
           "default_dtype 是 model.py 模块级全局变量（Transformer.__init__ "
           "884 赋值，见 ev_st_module_default_dtype），只被本自定义 Linear/"
           "Parallel 系列消费；Gate 参数不经此类（model.py:562 直接 "
           "nn.Parameter(torch.empty)，无显式 dtype → torch 默认 dtype）",
           "source_checked"),
        _E("ev_st_module_default_dtype", m,
           "Transformer.__init__ module-global default_dtype", 884, 884,
           'default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else '
           'torch.bfloat16',
           "Transformer.__init__ 只给模块全局 default_dtype 赋值（args.dtype="
           "\"fp8\" → fp8_e4m3fn）——普通模块变量（供 136 自定义 Linear 默认 "
           "dtype），不是 torch.set_default_dtype，不能改变 torch.empty 的"
           "默认 dtype；wq_a/wq_b/wkv/wo_b/main_proj/shared_experts 等经"
           "自定义 Linear 的真实量化权重按此机制 fp8（不改动）",
           "source_checked"),
        _E("ev_st_entry_default_dtype", m,
           "reference entry torch default dtype", 949, 950,
           'if __name__ == "__main__":\n'
           "    torch.set_default_dtype(torch.bfloat16)",
           "参考入口（__main__）显式 torch.set_default_dtype(torch.bfloat16)"
           " → torch 全局默认 dtype bf16 → Gate.weight（562 torch.empty 无"
           "显式 dtype）存储 bf16。Gate.forward（570）weight.float()/x."
           "float() 是 fp32 求值不是存储证据——参数存储/计算 dtype 分离。"
           "非参考入口若以其他 torch 默认 dtype 构建，Gate.weight 存储"
           "随之改变——该依赖显式声明于此，不从 args.dtype 推断",
           "source_checked"),
        _E("ev_st_expert_dtype", m, "MoE expert dtype", 628, 632,
           "expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == \"fp4\" else None\n"
           "self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype, swiglu_limit=args.swiglu_limit) if self.experts_start_idx <= i < self.experts_end_idx else None\n"
           "                               for i in range(self.n_routed_experts)])\n"
           "assert args.n_shared_experts == 1\n"
           "self.shared_experts = Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)",
           "专家参数 dtype：config expert_dtype=fp4 → 路由专家 weight "
           "fp4_e2m1fn_x2（[E,H//2] 逻辑 [E,H] + scale）、shared expert "
           "default dtype fp8；SwiGLU w1/w3 [E,D_in]、w2 [E,D_out] 按 "
           "moe_inter_dim=2048；n_shared_experts==1 assert", "located"),
        _E("ev_st_moe_route_causal", m, "MoE.forward routing", 636, 645,
           "weights, indices = self.gate(x, input_ids.flatten())\n"
           "y = torch.zeros_like(x, dtype=torch.float32)\n"
           "counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()\n"
           "for i in range(self.experts_start_idx, self.experts_end_idx):\n"
           "    if counts[i] == 0:\n"
           "        continue\n"
           "    expert = self.experts[i]\n"
           "    idx, top = torch.where(indices == i)\n"
           "    y[idx] += expert(x[idx], weights[idx, top, None])",
           "MoE 路由因果：counts（dispatch 消费 indices 产出）决定空专家"
           "跳过；(idx,top)=where(indices==i) 决定专家 i 的 token 子集与"
           "其权重槽位——expert(x[idx],weights[idx,top,None]) 二者都被"
           "消费。N_e=counts[i] 数据相关静态未知；stage 图以 dispatch→"
           "expert 的 select（token 子集）+ control（counts 门控）边表达",
           "source_checked"),
        _E("ev_st_stage_class", m, "DSparkBlock", 818, 820,
           "class DSparkBlock(Block):\n"
           '    """DSpark stage stored under the mtp.* checkpoint namespace."""\n'
           "    attention_cls = DSparkAttention",
           "DSpark stage 类：Block 子类、attention_cls 换 DSparkAttention、"
           "checkpoint 命名空间 mtp.*——Block 双 hc 前向骨架（hc_pre→norm→"
           "attn→hc_post→hc_pre→norm→MoE→hc_post）原样继承", "located"),
        _E("ev_st_stage_role", m, "DSparkBlock.__init__ role branches", 825, 841,
           "stage_id = layer_id - args.n_layers\n"
           "if stage_id == 0:\n"
           "    self.main_proj = Linear(args.dim * len(args.dspark_target_layer_ids), args.dim)\n"
           "    self.main_norm = RMSNorm(args.dim, args.norm_eps)\n"
           "if stage_id == args.n_mtp_layers - 1:\n"
           "    self.norm = RMSNorm(args.dim, args.norm_eps)\n"
           "    self.markov_head = DSparkMarkovHead(args.vocab_size, args.dspark_markov_rank)\n"
           "    self.confidence_head = DSparkConfidenceHead(args.dim + args.dspark_markov_rank)",
           "角色分支：stage_id==0 专有 main_proj（H×N_tgt→H）+main_norm；"
           "stage_id==n_mtp_layers-1（last）专有 norm+markov_head+"
           "confidence_head+hc_head 三参数（838-841）；中间 stage 无任何角色"
           "专有模块；stage_id = layer_id − n_layers（43）→ 实例 layer_id "
           "43/44/45", "source_checked"),
        _E("ev_st_layer_ids", m, "Transformer mtp construction", 898, 904,
           "self.mtp = torch.nn.ModuleList()\n"
           "self.target_layer_ids = args.dspark_target_layer_ids\n"
           "if args.dspark_block_size:\n"
           "    for layer_id in range(args.n_mtp_layers):\n"
           "        self.mtp.append(DSparkBlock(args.n_layers + layer_id, args))\n"
           "        self.mtp[-1].embed = self.embed",
           "三实例：DSparkBlock(n_layers+layer_id) → layer_id 43/44/45"
           "（layer constructor 编号，主干仍 43 层，非 46 个 main layers）；"
           "mtp.0/1/2 checkpoint 路径；embed/head 逐实例绑定 Transformer "
           "同一对象（903-904 全部实例 self.mtp[-1].embed = self.embed / "
           ".head = self.head），非三份复制权重", "source_checked"),
        _E("ev_st_forward_spec", m, "Transformer.forward_spec", 929, 935,
           "def forward_spec(self, input_ids: torch.Tensor, main_hidden: torch.Tensor, start_pos: int = 0):\n"
           "    h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)\n"
           "    for layer in self.mtp:\n"
           "        h = layer(h, start_pos, input_ids, main_x)\n"
           "    if start_pos == 0:\n"
           "        return",
           "链路：forward_embed 只在 stage0 调用（main_x 与草稿 h 同点产出）；"
           "同一 main_x 传入全部三 stage（第三实参逐层透传，非各 stage "
           "独立 readout）；h 逐 stage 串接 stage0→1→2；prefill（start_pos==0）"
           "在此早退——只构建三 stage cache，草稿输出与 head 不执行", "source_checked"),
        _E("ev_st_forward_embed", m, "DSparkBlock.forward_embed", 851, 861,
           "def forward_embed(self, main_hidden: torch.Tensor, input_ids: torch.Tensor):\n"
           "    assert self.embed is not None\n"
           "    main_x = self.main_norm(self.main_proj(main_hidden))\n"
           "    draft_input_ids = input_ids.new_full([input_ids.size(0), self.block_size], self.noise_token_id)\n"
           "    draft_input_ids[:, 0] = input_ids\n"
           "    x = self.embed(draft_input_ids)\n"
           "    x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)\n"
           "    return x, main_x",
           "first 入口（stage0 专有）：main_x = main_norm(main_proj(main_hidden))"
           "——共享 main_x 的唯一生产点；草稿 token = noise 填充 + 首位真实 "
           "input_ids；embed（Transformer 共享 embed）→ HC 展开 repeat 成 "
           "[B,N_draft,N_hc,H]；main_hidden 形状 [B,T,H×N_tgt]（readout concat）",
           "source_checked"),
        _E("ev_st_forward_head", m, "DSparkBlock.forward_head", 860, 874,
           "def forward_head(self, x: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:\n"
           "    assert self.head is not None\n"
           "    x = self.hc_head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)\n"
           "    logits = self.head(self.norm(x), full_logits=True)\n"
           "    output_ids = input_ids.new_empty(input_ids.size(0), self.block_size + 1)\n"
           "    for i in range(self.block_size):\n"
           "        logits_bias, markov_embed = self.markov_head(output_ids[:, i])\n"
           "    markov_embed = torch.stack(markov_embeds, dim=1)\n"
           "    confidence = self.confidence_head(x, markov_embed)",
           "last 出口（stage2 专有）：hc_head（N_hc 流→单流，sigmoid 门控）"
           "→ norm → head（Transformer 共享 head，full_logits）→ Markov 逐位"
           "采样循环（block_size 次 markov_head + sample）→ confidence_head；"
           "内部图本轮 pending，本条只锚定出口链存在与专有性", "located",
           note="采样循环逐位细节与 sample() 调用边界本轮未展开成图；"
                "forward_head 完整内部子图 pending"),
        _E("ev_st_readout", m, "Transformer.forward main readout", 917, 925,
           "main_hiddens = []\n"
           "for i, layer in enumerate(self.layers):\n"
           "    h = layer(h, start_pos, input_ids)\n"
           "    if i in self.target_layer_ids:\n"
           "        main_hiddens.append(h.mean(dim=2))",
           "readout 汇聚：主干层 i∈{40,41,42}（dspark_target_layer_ids）各 "
           "mean(hc 流，dim=2) 得 [B,T,H]，按 target_layer_ids 顺序 append；"
           "925 torch.cat(main_hiddens, dim=-1) → main_hidden [B,T,H×N_tgt]；"
           "顺序 = target_layer_ids 顺序（40,41,42），非按 stage 分配", "source_checked"),
        _E("ev_st_main_proj", m, "DSparkBlock.main_proj shape", 830, 833,
           "if stage_id == 0:\n"
           '    assert len(args.dspark_target_layer_ids) > 0, "DSpark needs target layers"\n'
           "    self.main_proj = Linear(args.dim * len(args.dspark_target_layer_ids), args.dim)\n"
           "    self.main_norm = RMSNorm(args.dim, args.norm_eps)",
           "main_proj 输入宽 = H × N_tgt（=H×3=12288）：消费有序 concat 的"
           "三个 readout；输出宽 H。N_tgt 轴依据；main_norm 是 stage0 专有",
           "located"),
        _E("ev_st_ratios_layout", rc, "compress_ratios layout", 39, 39,
           '"compress_ratios": [0, 0, 4, 128, 4,',
           "46 项布局 = 前 43 项主干层 0-42 消费（Block 层）+ 后 3 项 draft 层"
           " 43-45 消费（DSparkBlock 层，全 0，model.py:753 assert "
           "compress_ratio==0）；n_layers=43（run config:5）、n_mtp_layers=3"
           "（run config:6）。固定输入不修改", "located"),
        _E("ev_st_nextn", rootc, "num_nextn_predict_layers", 31, 31,
           '"num_nextn_predict_layers": 1,',
           "root config num_nextn_predict_layers=1：checkpoint nextn.* 单 MTP "
           "段（兼容 v3 MTPLoader 惯例）；与 run config n_mtp_layers=3 "
           "（inference model.py DSparkBlock 实例数，model.py:901）是不同字段"
           "不同消费者——前者 checkpoint 布局、后者推理图实例数", "located"),
        _E("ev_st_hc_head_stage2", m, "hc_head (stage last 专用调用)", 709, 716,
           "def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):\n"
           "    x = x.flatten(2).float()\n"
           "    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)\n"
           "    mixes = F.linear(x, hc_fn) * rsqrt\n"
           "    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps\n"
           "    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)",
           "hc_head：N_hc 流加权汇聚单流（sigmoid 门控，非 Sinkhorn）；"
           "参数 hc_head_fn[N_hc,N_hcH]/hc_head_base[N_hc]/hc_head_scale[1] "
           "只在 last stage 实例化（model.py:838-841）；主干侧另有同名方法用 "
           "Transformer.hc_head 三参数（model.py:908-910）——stage last "
           "专有参数，不与主干混淆", "located"),
        _E("ev_st_markov_head", m, "DSparkMarkovHead/DSparkConfidenceHead", 795, 815,
           "class DSparkMarkovHead(nn.Module):\n"
           "    self.markov_w1 = ParallelEmbedding(vocab_size, dspark_markov_rank)\n"
           "    self.markov_w2 = ParallelHead(vocab_size, dspark_markov_rank)\n"
           "class DSparkConfidenceHead(nn.Module):\n"
           "    self.proj = Linear(input_dim, 1, dtype=torch.float32)",
           "last 专有头参数形状：markov_w1 [V,Rm] embedding、markov_w2 "
           "[V//ws,Rm] lm_head、confidence proj [H+Rm,1]（fp32）；"
           "Rm=dspark_markov_rank=256。头内部图 pending，本条锚定参数形状"
           "与 last 专有性", "located"),
        _E("ev_st_pending_first_last", m, "first/last full subgraphs pending", None, None, None,
           "first 入口完整图（forward_embed 内部：embed 查表→HC 展开→"
           "hc_pre_attn 完整链）与 last 出口完整图（hc_head→norm→head "
           "full_logits→Markov 逐位采样→confidence）的节点级展开",
           "derived",
           derivation="本单范围声明：B2 只交付公共 stage 模板 + 三实例绑定 + "
                      "链路边界映射（model.py:929-935 拓扑事实）；first/last "
                      "完整内部图、Markov/Confidence 递推、全量集成由后续步骤"
                      "续派（本模块 pending 记录，不冒称已提取）"),
    )


def _axes(values: ModelValues) -> dict[str, NamedAxis]:
    """stage 图轴：与 B1 模块同源（同一 rules AXIS_DEFS 来源 + 同一组
    DSpark 新轴）+ 本模块两条骨架轴（HC3/C1）。无压缩轴（DSpark
    compress_ratio==0）。"""
    axes = _attn_axes(values)
    # HC3：hc_scale 三元组轴（model.py:677 torch.empty(3)，pre/post/comb
    # 三权重）；C1：hc_pre 归一化因子 keepdim 单例轴（model.py:683
    # mean(-1, keepdim=True) 的 size-1 尾轴，乘 N_mix 时 broadcast）。
    # 两条轴由 stage 骨架引入，B1 图不含 hc 链故不需要。
    axes["HC3"] = NamedAxis(
        "HC3", "hc_scale_triad",
        "hc_scale 三元组轴（pre/post/comb 三权重，torch.empty(3)）",
        "推导 model.py:677", "model.py:677", value=3)
    axes["C1"] = NamedAxis(
        "C1", "keepdim_singleton",
        "hc_pre 归一化因子 keepdim 单例轴（size 1，广播到 N_mix）",
        "推导 model.py:683", "model.py:683", value=1)
    return axes


# ---------------------------------------------------------------------------
# stage 图：公共 Block 双 hc 骨架 + 真实 DSparkAttention 子图 + TopK MoE
# ---------------------------------------------------------------------------

def _add_tensors(b: _GraphBuilder, axes: dict[str, NamedAxis]) -> dict[str, str]:
    """stage 张量角色 → stable_id（前缀 stage:）。

    四部分：公共参数（mHC/hc_scale/norm/gate——每实例独立身份，图记录
    模板身份）、Block 骨架级（hc 链/norm/MoE，model.py:695-707 真实控制
    流）、Attention 内部（_add_attn_tensors 前缀 stage:，与 B1 同一实现
    ——节点端口引用的张量就是这些 stage: 张量，边界映射三方一致）、
    prefill passthrough（DSparkBlock.forward 845-849）。"""
    # Attention 内部张量（含 main_x/slot_write/draft_pos0/win_cache 等边界
    # 可见张量）：与 B1 同一函数同前缀语义，此处前缀 stage:
    t = _add_attn_tensors(b, axes, prefix="stage",
                          draft_in_role="x_draft_single",
                          main_in_role="main_x")

    def add(role: str, dtype: str, syms: tuple[str, ...],
            evidence: tuple[str, ...] = ()) -> str:
        tid = b.tensor(f"stage:{role}", dtype, syms, evidence)
        t[role] = tid
        return tid

    # ===== 公共参数张量（模板身份；实例路径由展开绑定表承载）=====
    # mHC 参数形状/dtype 依 model.py:670-678（set_dtype fp32 块）；
    # gate 参数依 model.py:562/567（TopK 分支）。gate.weight 存储 dtype
    # = 参考入口 torch 默认 dtype bf16（model.py:950 set_default_dtype；
    # torch.empty 无显式 dtype）——Gate.forward 的 .float() 是 fp32 求值
    # 非存储；884 default_dtype 模块变量只供自定义 Linear（136），gate
    # 不经该机制。存储/计算 dtype 分离。
    add("hc_attn_fn_w", "fp32", ("N_mix", "N_hcH"), ("ev_st_hc_params",))
    add("hc_attn_scale_w", "fp32", ("HC3",), ("ev_st_hc_params",))
    add("hc_attn_base_w", "fp32", ("N_mix",), ("ev_st_hc_params",))
    add("hc_ffn_fn_w", "fp32", ("N_mix", "N_hcH"), ("ev_st_hc_params",))
    add("hc_ffn_scale_w", "fp32", ("HC3",), ("ev_st_hc_params",))
    add("hc_ffn_base_w", "fp32", ("N_mix",), ("ev_st_hc_params",))
    add("attn_norm_w", "fp32", ("H",), ("ev_st_norm_params",))
    add("ffn_norm_w", "fp32", ("H",), ("ev_st_norm_params",))
    add("gate_weight_w", "bf16", ("E", "H"), ("ev_st_gate_params",
                                              "ev_st_entry_default_dtype"))
    add("gate_bias_w", "fp32", ("E",), ("ev_st_gate_params",))

    # ===== Block 骨架级：hc_pre/post（attention 侧）=====
    # decode 输入 HC 流 [B,N_draft,N_hc,H]（forward_embed 输出 / 上一
    # stage hc_post 输出，model.py:930-932）
    add("x_hc_in", "bf16", ("B", "N_draft", "N_hc", "H"), ())
    add("x_hc_attn_flat", "fp32", ("B", "N_draft", "N_hcH"), ("ev_hc_pre",))
    # 归一化因子（model.py:683 rsqrt(mean(x²)+eps)，keepdim 尾轴 C1）
    add("attn_norm_factor", "fp32", ("B", "N_draft", "C1"), ("ev_st_hc_norm_factor",))
    add("attn_mixes", "fp32", ("B", "N_draft", "N_mix"), ("ev_hc_pre",))
    add("attn_pre", "fp32", ("B", "N_draft", "N_hc"), ("ev_hc_pre",))
    add("attn_post", "fp32", ("B", "N_draft", "N_hc"), ("ev_hc_pre",))
    add("attn_comb", "fp32", ("B", "N_draft", "hc_j", "hc_k"), ("ev_hc_post",))
    add("x_attn_in", "bf16", ("B", "N_draft", "H"), ("ev_hc_pre",))
    add("x_attn_post", "bf16", ("B", "N_draft", "N_hc", "H"), ("ev_hc_post",))
    # ===== hc_pre/post（ffn 侧）=====
    add("x_ffn_flat_hc", "fp32", ("B", "N_draft", "N_hcH"), ("ev_hc_pre",))
    add("ffn_norm_factor", "fp32", ("B", "N_draft", "C1"), ("ev_st_hc_norm_factor",))
    add("ffn_mixes", "fp32", ("B", "N_draft", "N_mix"), ("ev_hc_pre",))
    add("ffn_pre", "fp32", ("B", "N_draft", "N_hc"), ("ev_hc_pre",))
    add("ffn_post", "fp32", ("B", "N_draft", "N_hc"), ("ev_hc_pre",))
    add("ffn_comb", "fp32", ("B", "N_draft", "hc_j", "hc_k"), ("ev_hc_post",))
    add("x_ffn", "bf16", ("B", "N_draft", "H"), ("ev_hc_pre",))
    add("x_ffn_normed", "bf16", ("B", "N_draft", "H"), ("ev_attn_norm",))
    # ===== MoE（TopK gate）=====
    add("gate_scores", "fp32", ("B", "N_draft", "E"), ("ev_gate_forward",))
    add("gate_scores_biased", "fp32", ("B", "N_draft", "E"), ("ev_gate_forward",))
    add("gate_indices", "int64", ("B", "N_draft", "K"), ("ev_gate_forward",))
    add("gate_weights", "fp32", ("B", "N_draft", "K"), ("ev_gate_forward",))
    # 每专家 token 数：数据相关 unknown（轴 N_e，rules 静态未知轴）
    add("n_e", "int64", ("E",), ())
    # 路由因果（model.py:639-645）：counts 空专家跳过（control 门控）、
    # (idx,top)=where(indices==i) 选中 token 子集与权重槽位对——数据相关
    # 形状（轴 N_e 不定），dispatch 节点产出、专家模板消费。
    add("route_select", "int64", ("N_e",), ("ev_st_moe_route_causal",))
    add("route_slot", "fp32", ("N_e",), ("ev_st_moe_route_causal",))
    add("expert_out", "fp32", ("B", "N_draft", "H"), ("ev_moe_dispatch",))
    add("shared_out", "fp32", ("B", "N_draft", "H"), ("ev_moe_shared",))
    add("moe_out", "bf16", ("B", "N_draft", "H"), ("ev_moe_shared",))
    # ===== 输出 =====
    add("x_hc_out", "bf16", ("B", "N_draft", "N_hc", "H"), ())
    # prefill passthrough 输出（DSparkBlock.forward 849：prefill 只调 attn，
    # attn prefill 分支 769 return x 原样——HC 展开流直接返回）
    add("x_hc_prefill_out", "bf16", ("B", "N_draft", "N_hc", "H"), ())
    return t


def _add_flow(b: _GraphBuilder, t: dict[str, str]) -> tuple[str, ...]:
    """stage 流：Block 骨架（decode 全链 model.py:695-707 + prefill
    passthrough 845-849）+ Attention 内部流（_add_attn_flow 同源调用，
    src_op=subgraph_input、不含 attention 内部 prefill return）。

    返回 Attention 内部节点 ID 元组（_add_attn_flow 的创建记录，stage
    fold 组成员的当前图真实来源）。

    相位规则（B2 补修 01）：全部 decode 专用骨架运行时消费边挂可求值
    g_decode——prefill 只在 Block 级 passthrough/cache 构建（845-849 早
    退），双 mHC/FFN/MoE 链 prefill 不可达；main_x→共享 mainKV 两相位
    共享不挂 guard；参数身份节点静态存在，其到 decode-only 算子的消费
    边同样挂 g_decode（参数身份不等于计算边 prefill 运行）。不通过删边
    实现屏蔽。"""
    def node(sid, op, gr, inputs=(), outputs=(), ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.node(sid, op, gr, inputs, outputs, ev)

    def edge(sid, src, dst, role, kind="data", **kw):
        ev = kw.pop("ev", ())
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.edge(sid, src, dst, t[role], kind, **kw)

    # ===== Attention 内部（与 B1 单一实现同源；内部节点/边 ID 同 B1）=====
    # 三个输入源节点由 Block 骨架喂入（fed_sources：源节点带 x 输入端口，
    # 数据流从骨架经源节点进入 attention 内部——真实消费关系，非悬空）；
    # 角色名与 _add_attn_tensors 对应（main_x / x_draft_single）。
    attn_flow_nodes = _add_attn_flow(
        b, t, src_op="subgraph_input", include_prefill=False,
        fed_sources=True, main_in_role="main_x",
        draft_in_role="x_draft_single")

    # ===== 公共参数节点（模板身份；实例路径由展开绑定表承载）=====
    # 双 mHC 参数（每实例独立，model.py:670-678 set_dtype fp32 块）；
    # norm/gate 参数（RMSNorm fp32 / Gate TopK 分支）。参数节点到
    # decode-only 算子的消费边挂 g_decode。
    node("stage.hc_attn_fn_w", "hc_fn_param", "parameter_op",
         (), (("y", t["hc_attn_fn_w"]),), ("ev_st_hc_params",))
    node("stage.hc_attn_scale_w", "hc_scale_param", "parameter_op",
         (), (("y", t["hc_attn_scale_w"]),), ("ev_st_hc_params",))
    node("stage.hc_attn_base_w", "hc_base_param", "parameter_op",
         (), (("y", t["hc_attn_base_w"]),), ("ev_st_hc_params",))
    node("stage.hc_ffn_fn_w", "hc_fn_param", "parameter_op",
         (), (("y", t["hc_ffn_fn_w"]),), ("ev_st_hc_params",))
    node("stage.hc_ffn_scale_w", "hc_scale_param", "parameter_op",
         (), (("y", t["hc_ffn_scale_w"]),), ("ev_st_hc_params",))
    node("stage.hc_ffn_base_w", "hc_base_param", "parameter_op",
         (), (("y", t["hc_ffn_base_w"]),), ("ev_st_hc_params",))
    node("stage.attn_norm_w", "rms_norm_weight", "parameter_op",
         (), (("y", t["attn_norm_w"]),), ("ev_st_norm_params",))
    node("stage.ffn_norm_w", "rms_norm_weight", "parameter_op",
         (), (("y", t["ffn_norm_w"]),), ("ev_st_norm_params",))
    node("stage.gate_weight_w", "gate_weight", "parameter_op",
         (), (("y", t["gate_weight_w"]),), ("ev_st_gate_params",))
    node("stage.gate_bias_w", "gate_bias", "parameter_op",
         (), (("y", t["gate_bias_w"]),), ("ev_st_gate_params",))

    # ===== 图输入（Block 骨架）=====
    node("stage.x_hc_in_src", "graph_input", "tensor_op",
         (), (("y", t["x_hc_in"]),), ())
    node("stage.main_x_src", "graph_input", "tensor_op",
         (), (("y", t["main_x"]),), ())

    # ===== 真实 h 输入消费（decode）=====
    # forward_spec 931 h = layer(h, ...) → Block.forward 695-696：
    # residual = x; x, post, comb = self.hc_pre(x, ...)——h 进入首个
    # hc_pre_attn.flatten。prefill 走 Block 级 passthrough（849），不进
    # 双 mHC——消费边挂 g_decode。
    edge("e_st_hcin_hcpre", ("stage.x_hc_in_src", "y"),
         ("stage.hc_pre_attn.flatten", "x"), "x_hc_in",
         guard_ref="g_decode")

    # ===== hc_pre（attention 侧）=====
    node("stage.hc_pre_attn.flatten", "flatten", "tensor_op",
         (("x", t["x_hc_in"]),),
         (("x_flat", t["x_hc_attn_flat"]), ("residual", t["x_hc_in"])),
         ("ev_hc_pre",))
    # 归一化因子（model.py:683）：rsqrt(mean(flat²)+eps)，keepdim 单例尾轴
    node("stage.hc_pre_attn.factor", "rms_factor", "tensor_op",
         (("x", t["x_hc_attn_flat"]),), (("y", t["attn_norm_factor"]),),
         ("ev_st_hc_norm_factor",))
    edge("e_st_hcpre_a_factor", ("stage.hc_pre_attn.flatten", "x_flat"),
         ("stage.hc_pre_attn.factor", "x"), "x_hc_attn_flat",
         guard_ref="g_decode")
    # mix（model.py:684）：F.linear(x, hc_fn) * rsqrt——linear 节点消费
    # 权重与归一化因子端口，输出才是 mixes
    node("stage.hc_pre_attn.mix", "linear", "parameter_op",
         (("x", t["x_hc_attn_flat"]), ("w", t["hc_attn_fn_w"]),
          ("factor", t["attn_norm_factor"])),
         (("y", t["attn_mixes"]),),
         ("ev_hc_pre", "ev_st_hc_norm_factor"))
    edge("e_st_hcpre_a_flat", ("stage.hc_pre_attn.flatten", "x_flat"),
         ("stage.hc_pre_attn.mix", "x"), "x_hc_attn_flat",
         guard_ref="g_decode")
    edge("e_st_hcpre_a_fn_w", ("stage.hc_attn_fn_w", "y"),
         ("stage.hc_pre_attn.mix", "w"), "hc_attn_fn_w",
         guard_ref="g_decode")
    edge("e_st_hcpre_a_factor_in", ("stage.hc_pre_attn.factor", "y"),
         ("stage.hc_pre_attn.mix", "factor"), "attn_norm_factor",
         guard_ref="g_decode")
    # sinkhorn（model.py:685）：opaque kernel，接口 (mixes, scale, base)
    node("stage.hc_pre_attn.sinkhorn", "sinkhorn_split", "opaque_kernel",
         (("mixes", t["attn_mixes"]), ("scale", t["hc_attn_scale_w"]),
          ("base", t["hc_attn_base_w"])),
         (("pre", t["attn_pre"]), ("post", t["attn_post"]),
          ("comb", t["attn_comb"])),
         ("ev_hc_pre", "ev_hc_sinkhorn_kernel"))
    edge("e_st_hcpre_a_mix", ("stage.hc_pre_attn.mix", "y"),
         ("stage.hc_pre_attn.sinkhorn", "mixes"), "attn_mixes",
         guard_ref="g_decode")
    edge("e_st_hcpre_a_scale_w", ("stage.hc_attn_scale_w", "y"),
         ("stage.hc_pre_attn.sinkhorn", "scale"), "hc_attn_scale_w",
         guard_ref="g_decode")
    edge("e_st_hcpre_a_base_w", ("stage.hc_attn_base_w", "y"),
         ("stage.hc_pre_attn.sinkhorn", "base"), "hc_attn_base_w",
         guard_ref="g_decode")
    node("stage.hc_pre_attn.reduce", "weighted_sum", "tensor_op",
         (("pre", t["attn_pre"]), ("x", t["x_hc_in"])),
         (("y", t["x_attn_in"]),), ("ev_hc_pre",))
    edge("e_st_hcpre_a_pre", ("stage.hc_pre_attn.sinkhorn", "pre"),
         ("stage.hc_pre_attn.reduce", "pre"), "attn_pre",
         guard_ref="g_decode")
    edge("e_st_hcpre_a_x", ("stage.hc_pre_attn.flatten", "residual"),
         ("stage.hc_pre_attn.reduce", "x"), "x_hc_in",
         guard_ref="g_decode")

    # ===== attn_norm → attention 输入（decode；Block.forward 697-699）=====
    node("stage.attn_norm", "rms_norm", "parameter_op",
         (("x", t["x_attn_in"]), ("w", t["attn_norm_w"])),
         (("y", t["x_draft_single"]),),
         ("ev_attn_norm", "ev_st_norm_params"))
    edge("e_st_hcpre_a_out", ("stage.hc_pre_attn.reduce", "y"),
         ("stage.attn_norm", "x"), "x_attn_in", guard_ref="g_decode")
    edge("e_st_attnnorm_w", ("stage.attn_norm_w", "y"),
         ("stage.attn_norm", "w"), "attn_norm_w", guard_ref="g_decode")
    # attn_norm 输出 = attention decode x（同一张量 dspark 角色名
    # x_draft_single——Block.forward 697-699 x = self.attn_norm(x) 后
    # self.attn(x, ...) 直接消费，同一存储非拷贝）；消费边从 attn_norm
    # 到内部源节点（fed_sources：源节点 x 端口），挂 g_decode。
    edge("e_st_norm_xsrc", ("stage.attn_norm", "y"),
         ("dspark.x_draft_single_src", "x"), "x_draft_single",
         guard_ref="g_decode")
    # main_x 骨架输入 → attention main 路径源节点（两相位共享，不挂 guard）
    edge("e_st_mainx_xsrc", ("stage.main_x_src", "y"),
         ("dspark.x_main_in_src", "x"), "main_x")

    # ===== hc_post（attention 残差至 post；Block.forward 700）=====
    node("stage.hc_post_attn.combine", "hc_combine", "tensor_op",
         (("x", t["attn_out"]), ("post", t["attn_post"]),
          ("comb", t["attn_comb"]), ("residual", t["x_hc_in"])),
         (("y", t["x_attn_post"]),), ("ev_hc_post",))
    edge("e_st_attn_out_post", ("dspark.wo_b", "y"),
         ("stage.hc_post_attn.combine", "x"), "attn_out",
         guard_ref="g_decode")
    edge("e_st_hcpost_a_post", ("stage.hc_pre_attn.sinkhorn", "post"),
         ("stage.hc_post_attn.combine", "post"), "attn_post",
         guard_ref="g_decode")
    edge("e_st_hcpost_a_comb", ("stage.hc_pre_attn.sinkhorn", "comb"),
         ("stage.hc_post_attn.combine", "comb"), "attn_comb",
         guard_ref="g_decode")
    edge("e_st_hcpost_a_residual", ("stage.hc_pre_attn.flatten", "residual"),
         ("stage.hc_post_attn.combine", "residual"), "x_hc_in",
         guard_ref="g_decode")

    # ===== hc_pre（ffn 侧；Block.forward 702）=====
    node("stage.hc_pre_ffn.flatten", "flatten", "tensor_op",
         (("x", t["x_attn_post"]),),
         (("x_flat", t["x_ffn_flat_hc"]), ("residual", t["x_attn_post"])),
         ("ev_hc_pre",))
    edge("e_st_hcpost_a_out", ("stage.hc_post_attn.combine", "y"),
         ("stage.hc_pre_ffn.flatten", "x"), "x_attn_post",
         guard_ref="g_decode")
    # 归一化因子（ffn 侧，同 model.py:683）
    node("stage.hc_pre_ffn.factor", "rms_factor", "tensor_op",
         (("x", t["x_ffn_flat_hc"]),), (("y", t["ffn_norm_factor"]),),
         ("ev_st_hc_norm_factor",))
    edge("e_st_hcpre_f_factor", ("stage.hc_pre_ffn.flatten", "x_flat"),
         ("stage.hc_pre_ffn.factor", "x"), "x_ffn_flat_hc",
         guard_ref="g_decode")
    node("stage.hc_pre_ffn.mix", "linear", "parameter_op",
         (("x", t["x_ffn_flat_hc"]), ("w", t["hc_ffn_fn_w"]),
          ("factor", t["ffn_norm_factor"])),
         (("y", t["ffn_mixes"]),),
         ("ev_hc_pre", "ev_st_hc_norm_factor"))
    edge("e_st_hcpre_f_flat", ("stage.hc_pre_ffn.flatten", "x_flat"),
         ("stage.hc_pre_ffn.mix", "x"), "x_ffn_flat_hc",
         guard_ref="g_decode")
    edge("e_st_hcpre_f_fn_w", ("stage.hc_ffn_fn_w", "y"),
         ("stage.hc_pre_ffn.mix", "w"), "hc_ffn_fn_w",
         guard_ref="g_decode")
    edge("e_st_hcpre_f_factor_in", ("stage.hc_pre_ffn.factor", "y"),
         ("stage.hc_pre_ffn.mix", "factor"), "ffn_norm_factor",
         guard_ref="g_decode")
    node("stage.hc_pre_ffn.sinkhorn", "sinkhorn_split", "opaque_kernel",
         (("mixes", t["ffn_mixes"]), ("scale", t["hc_ffn_scale_w"]),
          ("base", t["hc_ffn_base_w"])),
         (("pre", t["ffn_pre"]), ("post", t["ffn_post"]),
          ("comb", t["ffn_comb"])),
         ("ev_hc_pre", "ev_hc_sinkhorn_kernel"))
    edge("e_st_hcpre_f_mix", ("stage.hc_pre_ffn.mix", "y"),
         ("stage.hc_pre_ffn.sinkhorn", "mixes"), "ffn_mixes",
         guard_ref="g_decode")
    edge("e_st_hcpre_f_scale_w", ("stage.hc_ffn_scale_w", "y"),
         ("stage.hc_pre_ffn.sinkhorn", "scale"), "hc_ffn_scale_w",
         guard_ref="g_decode")
    edge("e_st_hcpre_f_base_w", ("stage.hc_ffn_base_w", "y"),
         ("stage.hc_pre_ffn.sinkhorn", "base"), "hc_ffn_base_w",
         guard_ref="g_decode")
    node("stage.hc_pre_ffn.reduce", "weighted_sum", "tensor_op",
         (("pre", t["ffn_pre"]), ("x", t["x_attn_post"])),
         (("y", t["x_ffn"]),), ("ev_hc_pre",))
    edge("e_st_hcpre_f_pre", ("stage.hc_pre_ffn.sinkhorn", "pre"),
         ("stage.hc_pre_ffn.reduce", "pre"), "ffn_pre",
         guard_ref="g_decode")
    edge("e_st_hcpre_f_x", ("stage.hc_pre_ffn.flatten", "residual"),
         ("stage.hc_pre_ffn.reduce", "x"), "x_attn_post",
         guard_ref="g_decode")

    # ===== ffn_norm → TopK MoE（Block.forward 703-704）=====
    node("stage.ffn_norm", "rms_norm", "parameter_op",
         (("x", t["x_ffn"]), ("w", t["ffn_norm_w"])),
         (("y", t["x_ffn_normed"]),),
         ("ev_attn_norm", "ev_st_norm_params"))
    edge("e_st_hcpre_f_out", ("stage.hc_pre_ffn.reduce", "y"),
         ("stage.ffn_norm", "x"), "x_ffn", guard_ref="g_decode")
    edge("e_st_ffnnorm_w", ("stage.ffn_norm_w", "y"),
         ("stage.ffn_norm", "w"), "ffn_norm_w", guard_ref="g_decode")
    # MoE TopK Gate（参考提取公共结构复用：score→bias→topk→gather→dispatch
    # →专家模板→shared expert→accumulate，与 deepseek_v4_reference._add_moe
    # 同构；sqrtsoftplus 计分，model.py:569-589；stage 层 43/44/45 ≥
    # n_hash_layers=3 → 全 TopK，无 hash 查表分支）。gate 权重/偏置参数
    # 真实消费（model.py:562/567 → 570 linear 调用）。
    edge("e_st_ffnnorm_gate", ("stage.ffn_norm", "y"),
         ("stage.moe.gate.score", "x"), "x_ffn_normed", guard_ref="g_decode")
    node("stage.moe.gate.score", "moe_gate_score", "parameter_op",
         (("x", t["x_ffn_normed"]), ("w", t["gate_weight_w"])),
         (("scores", t["gate_scores"]),),
         ("ev_gate_forward", "ev_st_gate_params"))
    edge("e_st_gatew_score", ("stage.gate_weight_w", "y"),
         ("stage.moe.gate.score", "w"), "gate_weight_w",
         guard_ref="g_decode")
    node("stage.moe.gate.bias", "add_bias", "parameter_op",
         (("x", t["gate_scores"]), ("b", t["gate_bias_w"])),
         (("y", t["gate_scores_biased"]),), ("ev_gate_forward",))
    edge("e_st_score_bias", ("stage.moe.gate.score", "scores"),
         ("stage.moe.gate.bias", "x"), "gate_scores", guard_ref="g_decode")
    edge("e_st_gateb_bias", ("stage.gate_bias_w", "y"),
         ("stage.moe.gate.bias", "b"), "gate_bias_w", guard_ref="g_decode")
    node("stage.moe.gate.topk", "topk_select", "index_op",
         (("x", t["gate_scores_biased"]),), (("indices", t["gate_indices"]),),
         ("ev_gate_forward",))
    edge("e_st_bias_topk", ("stage.moe.gate.bias", "y"),
         ("stage.moe.gate.topk", "x"), "gate_scores_biased",
         guard_ref="g_decode")
    node("stage.moe.gate.gather", "gather_weights", "tensor_op",
         (("original", t["gate_scores"]), ("indices", t["gate_indices"])),
         (("weights", t["gate_weights"]),),
         ("ev_gate_forward", "ev_weight_norm"))
    edge("e_st_orig_gather", ("stage.moe.gate.score", "scores"),
         ("stage.moe.gate.gather", "original"), "gate_scores",
         guard_ref="g_decode")
    edge("e_st_idx_gather", ("stage.moe.gate.topk", "indices"),
         ("stage.moe.gate.gather", "indices"), "gate_indices",
         guard_ref="g_decode")
    # 关键结构事实（与参考图同源）：偏置不进 gather（选择偏置不影响权重）
    # dispatch（model.py:639,643）：消费 indices 产出两个路由事实——counts
    # （bincount，641 counts==0 continue 空专家跳过的门控）与 routing
    # （643 (idx,top)=torch.where(indices==i) 的选中 token 子集+权重槽位
    # 对）。N_e 数据相关静态未知（rules 轴），不编造数值。
    node("stage.moe.dispatch", "token_dispatch", "index_op",
         (("indices", t["gate_indices"]),),
         (("counts", t["n_e"]), ("routing", t["route_select"])),
         ("ev_moe_dispatch", "ev_st_moe_route_causal"))
    edge("e_st_idx_dispatch", ("stage.moe.gate.topk", "indices"),
         ("stage.moe.dispatch", "indices"), "gate_indices",
         guard_ref="g_decode")
    # 路由因果到专家（model.py:642-645）：counts==0 空专家跳过（control
    # 门控）；(idx,top) 决定专家 i 的 token 子集（select）与权重槽位
    # （weights[idx,top,None]）——专家模板二者真实消费。
    node("stage.moe.expert_template", "swiglu_expert", "parameter_op",
         (("x", t["x_ffn_normed"]), ("weights", t["gate_weights"]),
          ("routing", t["route_select"]), ("gate_counts", t["n_e"])),
         (("y", t["expert_out"]),), ("ev_expert", "ev_moe_dispatch",
                                     "ev_st_moe_route_causal"))
    edge("e_st_gate_expert", ("stage.ffn_norm", "y"),
         ("stage.moe.expert_template", "x"), "x_ffn_normed",
         guard_ref="g_decode")
    edge("e_st_w_expert", ("stage.moe.gate.gather", "weights"),
         ("stage.moe.expert_template", "weights"), "gate_weights",
         guard_ref="g_decode")
    edge("e_st_dispatch_select", ("stage.moe.dispatch", "routing"),
         ("stage.moe.expert_template", "routing"), "route_select",
         kind="select", guard_ref="g_decode",
         ev=("ev_st_moe_route_causal",))
    edge("e_st_dispatch_gate", ("stage.moe.dispatch", "counts"),
         ("stage.moe.expert_template", "gate_counts"), "n_e",
         kind="control", guard_ref="g_decode",
         ev=("ev_st_moe_route_causal",))
    node("stage.moe.shared_experts", "swiglu_expert", "parameter_op",
         (("x", t["x_ffn_normed"]),), (("y", t["shared_out"]),),
         ("ev_moe_shared",))
    edge("e_st_gate_shared", ("stage.ffn_norm", "y"),
         ("stage.moe.shared_experts", "x"), "x_ffn_normed",
         guard_ref="g_decode")
    node("stage.moe.out", "accumulate_add", "tensor_op",
         (("routed", t["expert_out"]), ("shared", t["shared_out"])),
         (("y", t["moe_out"]),), ("ev_moe_shared",))
    edge("e_st_expert_acc", ("stage.moe.expert_template", "y"),
         ("stage.moe.out", "routed"), "expert_out", guard_ref="g_decode")
    edge("e_st_shared_acc", ("stage.moe.shared_experts", "y"),
         ("stage.moe.out", "shared"), "shared_out", guard_ref="g_decode")

    # ===== hc_post（ffn 残差至 post；Block.forward 705）=====
    node("stage.hc_post_ffn.combine", "hc_combine", "tensor_op",
         (("x", t["moe_out"]), ("post", t["ffn_post"]),
          ("comb", t["ffn_comb"]), ("residual", t["x_attn_post"])),
         (("y", t["x_hc_out"]),), ("ev_hc_post",))
    edge("e_st_moe_post", ("stage.moe.out", "y"),
         ("stage.hc_post_ffn.combine", "x"), "moe_out", guard_ref="g_decode")
    edge("e_st_hcpost_f_post", ("stage.hc_pre_ffn.sinkhorn", "post"),
         ("stage.hc_post_ffn.combine", "post"), "ffn_post",
         guard_ref="g_decode")
    edge("e_st_hcpost_f_comb", ("stage.hc_pre_ffn.sinkhorn", "comb"),
         ("stage.hc_post_ffn.combine", "comb"), "ffn_comb",
         guard_ref="g_decode")
    edge("e_st_hcpost_f_residual", ("stage.hc_pre_ffn.flatten", "residual"),
         ("stage.hc_post_ffn.combine", "residual"), "x_attn_post",
         guard_ref="g_decode")

    # ===== prefill passthrough（DSparkBlock.forward 845-849）=====
    # prefill 只调 attn（cache 构建），不走 Block 双 hc 前向；attn prefill
    # 分支 return x 原样（769）——HC 展开流 passthrough 返回。消费边挂
    # g_prefill。
    node("stage.prefill_passthrough", "passthrough", "tensor_op",
         (("x", t["x_hc_in"]),), (("y", t["x_hc_prefill_out"]),),
         ("ev_ds_block_prefill",))
    edge("e_st_prefill_in", ("stage.x_hc_in_src", "y"),
         ("stage.prefill_passthrough", "x"), "x_hc_in",
         guard_ref="g_prefill")
    return attn_flow_nodes


def _attn_boundary_mappings(t: dict[str, str]) -> tuple[FoldBoundaryMapping, ...]:
    """stage Attention 折叠组的边界映射：只有真实穿越折叠边界的端口。
    内部 KV 写（cache_update.kv）与坐标消费（cache_update.slot）都在组内
    完成，不产生边界映射；真正的边界：
    - input x：attn_norm 输出 → 内部源节点 dspark.x_draft_single_src.x
      （decode x 数据流入处，fed 端口）；
    - input main_x：main_x 骨架输入 → dspark.x_main_in_src.x；
    - output y：内部 wo_b.y（attn_out）→ 骨架 hc_post combine；
    - output cache（state）：win_cache 状态版本（cache_update.cache_full）
      → 本实例独立 state_id 存储。
    internal 节点全部是 fold 组成员（B1 同源节点），公开携带、可序列化
    往返，不丢内部连线。"""
    return (
        FoldBoundaryMapping(
            direction="input", external_port="x", edge_kind="data",
            internal_node_id="dspark.x_draft_single_src",
            internal_port="x",
            evidence_refs=("ev_ds_draft_q", "ev_st_pending_first_last")),
        FoldBoundaryMapping(
            direction="input", external_port="main_x", edge_kind="data",
            internal_node_id="dspark.x_main_in_src", internal_port="x",
            evidence_refs=("ev_ds_main_kv", "ev_st_forward_spec")),
        FoldBoundaryMapping(
            direction="output", external_port="y", edge_kind="data",
            internal_node_id="dspark.wo_b", internal_port="y",
            evidence_refs=("ev_ds_ogroup",)),
        FoldBoundaryMapping(
            direction="output", external_port="cache", edge_kind="state",
            internal_node_id="dspark.cache_update",
            internal_port="cache_full",
            evidence_refs=("ev_ds_prefill", "ev_ds_decode")),
    )


# ---------------------------------------------------------------------------
# 绑定与链路图
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageBinding:
    """一个真实 DSparkBlock 实例的绑定记录（纯角色逻辑派生，不改配置）。

    - stage_id：0-based stage 序号（DSparkBlock stage_id = layer_id −
      n_layers，model.py:825）；
    - actual_layer_id：layer constructor 编号 = n_main_layers + stage_id
      （model.py:902）——主干仍 43 层，本编号只用于 constructor；
    - param_path：checkpoint 参数路径前缀（mtp.{stage_id}.*）；
    - state_id：本实例 win_cache 独立状态 ID；
    - is_first/is_last：由数量纯逻辑计算（single → 同时 true）；
    - template_id：公共模板 ID（三实例同源）；
    - role_modules：角色专有模块清单（first：main_proj/main_norm/
      forward_embed；last：norm/markov_head/confidence_head/forward_head/
      hc_head 参数；middle：无；single：并集）；
    - pending_subgraphs：本实例 pending 的子图说明（角色专有模块不冒称
      已提取）。
    """

    stage_id: int
    actual_layer_id: int
    param_path: str
    state_id: str
    is_first: bool
    is_last: bool
    template_id: str
    role_modules: tuple[str, ...]
    pending_subgraphs: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "actual_layer_id": self.actual_layer_id,
            "param_path": self.param_path,
            "state_id": self.state_id,
            "is_first": self.is_first,
            "is_last": self.is_last,
            "template_id": self.template_id,
            "role_modules": list(self.role_modules),
            "pending_subgraphs": list(self.pending_subgraphs),
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "StageBinding":
        if not isinstance(doc, dict):
            raise ExtractionError("StageBinding 记录必须是 JSON 对象")
        required = ("stage_id", "actual_layer_id", "param_path", "state_id",
                    "is_first", "is_last", "template_id", "role_modules",
                    "pending_subgraphs")
        missing = [k for k in required if k not in doc]
        if missing:
            raise ExtractionError(f"StageBinding 缺字段: {missing}")
        sid = doc["stage_id"]
        if not isinstance(sid, int) or isinstance(sid, bool) or sid < 0:
            raise ExtractionError(f"stage_id 必须是非负 int，得到 {sid!r}")
        for f in ("role_modules", "pending_subgraphs"):
            if not isinstance(doc[f], list):
                raise ExtractionError(f"{f} 必须是 JSON list")
        obj = cls(
            stage_id=sid, actual_layer_id=doc["actual_layer_id"],
            param_path=doc["param_path"], state_id=doc["state_id"],
            is_first=doc["is_first"], is_last=doc["is_last"],
            template_id=doc["template_id"],
            role_modules=tuple(doc["role_modules"]),
            pending_subgraphs=tuple(doc["pending_subgraphs"]),
        )
        for f in ("param_path", "state_id", "template_id"):
            v = getattr(obj, f)
            if not isinstance(v, str) or not v.strip():
                raise ExtractionError(f"{f} 必须是非空字符串")
        for f in ("is_first", "is_last"):
            if not isinstance(getattr(obj, f), bool):
                raise ExtractionError(f"{f} 必须是布尔")
        if not isinstance(obj.actual_layer_id, int) or isinstance(
                obj.actual_layer_id, bool):
            raise ExtractionError("actual_layer_id 必须是 int")
        for f in ("role_modules", "pending_subgraphs"):
            v = getattr(obj, f)
            if not isinstance(v, tuple) or not all(
                    isinstance(s, str) and s.strip() for s in v):
                raise ExtractionError(f"{f} 必须是非空字符串的序列")
        return obj


def derive_stage_bindings(n_stages: int, n_main_layers: int,
                          template_id: str = STAGE_TEMPLATE_ID
                          ) -> tuple[StageBinding, ...]:
    """真实实例绑定（纯角色逻辑；不修改固定配置）。

    actual_layer_id = n_main_layers + stage_id（model.py:902
    DSparkBlock(args.n_layers + layer_id)）；首/末角色由数量计算：
    i==0 → first；i==n_stages-1 → last；n_stages==1 时同一实例同时
    first 与 last（纯逻辑分支覆盖）。role_modules 按 model.py:830-841
    真实分支记录；middle 无角色专有模块（只有共享 embed/head 引用）。"""
    if n_stages < 1:
        raise ExtractionError(f"n_stages 必须 ≥1，得到 {n_stages}")
    if n_main_layers < 1:
        raise ExtractionError(f"n_main_layers 必须 ≥1，得到 {n_main_layers}")
    shared = ("shared.embed", "shared.head")
    bindings: list[StageBinding] = []
    for i in range(n_stages):
        single = (i == 0 == n_stages - 1)
        first_only = i == 0
        last_only = i == n_stages - 1
        if single:
            role = ("main_proj", "main_norm", "forward_embed",
                    "norm", "markov_head", "confidence_head", "forward_head",
                    "hc_head_fn", "hc_head_base", "hc_head_scale") + shared
        elif first_only:
            role = ("main_proj", "main_norm", "forward_embed") + shared
        elif last_only:
            role = ("norm", "markov_head", "confidence_head", "forward_head",
                    "hc_head_fn", "hc_head_base", "hc_head_scale") + shared
        else:
            role = shared
        pending: list[str] = []
        if first_only or single:
            pending.append(
                "first 入口完整图（forward_embed 内部：main_proj/main_norm→"
                "noise 填充→shared.embed 查表→HC 展开 repeat）pending")
        if last_only or single:
            pending.append(
                "last 出口完整图（hc_head→norm→shared.head full_logits→"
                "Markov 逐位采样→confidence_head）pending")
        bindings.append(StageBinding(
            stage_id=i,
            actual_layer_id=n_main_layers + i,
            param_path=f"mtp.{i}",
            state_id=f"dspark:win_cache:mtp.{i}",
            is_first=first_only or single,
            is_last=last_only or single,
            template_id=template_id,
            role_modules=role,
            pending_subgraphs=tuple(pending),
        ))
    return tuple(bindings)


# ---------------------------------------------------------------------------
# 参数/状态展开绑定表（task_b2 展开关系交付：公共 template 实体 → 各
# mtp.i 实例实体，真实 shape/dtype/证据锚定，可规范 JSON 往返与读回
# 一致性核对）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemplateParamSpec:
    """公共模板的一个参数源/算子参数槽（template 实体）。

    - spec_id：模板局部稳定 ID（如 "stage.hc_attn_fn_w"）；
    - param_slot：DSparkBlock/Transformer 模块参数名（checkpoint 相对
      路径，如 "hc_attn_fn"）或 None（非参数槽的算子输入槽引用）；
    - shape：真实参数形状（ModelValues 派生数值元组，空 = 标量/无形状）；
    - dtype：真实参数 dtype（模型代码创建 dtype，非 checkpoint 存储
      dtype——差异在 note 声明）；
    - evidence：锚定证据 ID；
    - note：语义说明（消费点、实例独立/共享、checkpoint 存储差异）。
    """

    spec_id: str
    param_slot: str | None
    shape: tuple[int, ...]
    dtype: str
    evidence: tuple[str, ...]
    note: str

    def to_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "param_slot": self.param_slot,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "evidence": list(self.evidence),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "TemplateParamSpec":
        if not isinstance(doc, dict):
            raise ExtractionError("TemplateParamSpec 必须是 JSON 对象")
        for k in ("spec_id", "param_slot", "shape", "dtype", "evidence",
                  "note"):
            if k not in doc:
                raise ExtractionError(f"TemplateParamSpec 缺字段: {k}")
        shape = doc["shape"]
        if not isinstance(shape, list) or not all(
                isinstance(v, int) and not isinstance(v, bool)
                for v in shape):
            raise ExtractionError("TemplateParamSpec.shape 必须是 int 列表")
        ev = doc["evidence"]
        if not isinstance(ev, list) or not ev or not all(
                isinstance(s, str) and s.strip() for s in ev):
            raise ExtractionError(
                "TemplateParamSpec.evidence 必须是非空字符串非空列表")
        for k in ("spec_id", "dtype", "note"):
            if not isinstance(doc[k], str) or not doc[k].strip():
                raise ExtractionError(f"TemplateParamSpec.{k} 必须是非空字符串")
        slot = doc["param_slot"]
        if slot is not None and (not isinstance(slot, str) or not slot.strip()):
            raise ExtractionError(
                "TemplateParamSpec.param_slot 必须是非空字符串或 null")
        return cls(spec_id=doc["spec_id"], param_slot=slot,
                   shape=tuple(shape), dtype=doc["dtype"],
                   evidence=tuple(ev), note=doc["note"])


@dataclass(frozen=True)
class InstanceParamBinding:
    """一个模板参数源在 mtp.{i} 实例上的展开（instance 实体）。

    - spec_id：对应 TemplateParamSpec.spec_id（模板实体身份）；
    - stage_id：实例序号；
    - instance_path：checkpoint 完整参数路径（mtp.{i}.{param_slot}；
      shared 引用记 shared.embed/shared.head——Transformer 同一对象）；
    - shape/dtype：与模板核对一致（读回时逐项复检）。
    """

    spec_id: str
    stage_id: int
    instance_path: str
    shape: tuple[int, ...]
    dtype: str

    def to_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "stage_id": self.stage_id,
            "instance_path": self.instance_path,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "InstanceParamBinding":
        if not isinstance(doc, dict):
            raise ExtractionError("InstanceParamBinding 必须是 JSON 对象")
        for k in ("spec_id", "stage_id", "instance_path", "shape", "dtype"):
            if k not in doc:
                raise ExtractionError(f"InstanceParamBinding 缺字段: {k}")
        sid = doc["stage_id"]
        if not isinstance(sid, int) or isinstance(sid, bool) or sid < 0:
            raise ExtractionError(
                f"InstanceParamBinding.stage_id 必须是非负 int，得到 {sid!r}")
        shape = doc["shape"]
        if not isinstance(shape, list) or not all(
                isinstance(v, int) and not isinstance(v, bool)
                for v in shape):
            raise ExtractionError(
                "InstanceParamBinding.shape 必须是 int 列表")
        for k in ("spec_id", "instance_path", "dtype"):
            if not isinstance(doc[k], str) or not doc[k].strip():
                raise ExtractionError(
                    f"InstanceParamBinding.{k} 必须是非空字符串")
        return cls(spec_id=doc["spec_id"], stage_id=sid,
                   instance_path=doc["instance_path"], shape=tuple(shape),
                   dtype=doc["dtype"])


@dataclass(frozen=True)
class InstanceStateBinding:
    """一个模板 cache 状态在 mtp.{i} 实例上的独立身份。

    - template_state_id：公共模板 state_id（B1 _states 单一实现的
      "dspark:win_cache"——图状态注册 ID，非张量名）；
    - state_id：实例独立 state_id（"dspark:win_cache:mtp.{i}"）；
    - cache_tensor：模板 stage 图内 cache 张量的 name（"stage:win_cache"；
      图内 stable ID = "tensor:" + name，即 "tensor:stage:win_cache"——
      本字段记 name，stable ID 由张量命名规则确定）；
    - version_labels：模板版本对（v_hist → v_after，_states 默认两版本）；
    - transition_id：模板 transition（tr_dspark_cache）；
    - layer_id：实例 layer constructor 编号（43+i）。
    """

    stage_id: int
    template_state_id: str
    state_id: str
    cache_tensor: str
    version_labels: tuple[str, str]
    transition_id: str
    layer_id: int

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "template_state_id": self.template_state_id,
            "state_id": self.state_id,
            "cache_tensor": self.cache_tensor,
            "version_labels": list(self.version_labels),
            "transition_id": self.transition_id,
            "layer_id": self.layer_id,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "InstanceStateBinding":
        if not isinstance(doc, dict):
            raise ExtractionError("InstanceStateBinding 必须是 JSON 对象")
        for k in ("stage_id", "template_state_id", "state_id",
                  "cache_tensor", "version_labels", "transition_id",
                  "layer_id"):
            if k not in doc:
                raise ExtractionError(f"InstanceStateBinding 缺字段: {k}")
        sid = doc["stage_id"]
        lid = doc["layer_id"]
        for v, nm in ((sid, "stage_id"), (lid, "layer_id")):
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ExtractionError(
                    f"InstanceStateBinding.{nm} 必须是非负 int，得到 {v!r}")
        vl = doc["version_labels"]
        if (not isinstance(vl, list) or len(vl) != 2 or not all(
                isinstance(s, str) and s.strip() for s in vl)):
            raise ExtractionError(
                "InstanceStateBinding.version_labels 必须是二元字符串列表")
        for k in ("template_state_id", "state_id", "cache_tensor",
                  "transition_id"):
            if not isinstance(doc[k], str) or not doc[k].strip():
                raise ExtractionError(
                    f"InstanceStateBinding.{k} 必须是非空字符串")
        return cls(stage_id=sid, template_state_id=doc["template_state_id"],
                   state_id=doc["state_id"], cache_tensor=doc["cache_tensor"],
                   version_labels=(vl[0], vl[1]),
                   transition_id=doc["transition_id"], layer_id=lid)


@dataclass(frozen=True)
class StageExpansion:
    """参数/状态展开绑定表：公共 template 实体 → 各 mtp.i 实例实体。

    - param_specs：模板参数源清单（真实 shape/dtype/证据，涵盖 Attention
      参数、公共双 mHC、norm、TopK MoE gate/expert/shared expert、first/
      last 角色专有参数）；
    - param_bindings：每个 spec × 每个实例的展开（instance_path）；
    - state_bindings：模板 cache 状态 → 各实例独立身份（state_id/版本对/
      transition）；
    - shared_refs：共享 embed/head（只引用 Transformer 身份，不展开实例
      路径——model.py:903-904 同一对象）；
    - pending_slots：首末专有参数中内部图仍 pending 的槽位说明。
    """

    n_stages: int
    n_main_layers: int
    template_id: str
    param_specs: tuple[TemplateParamSpec, ...]
    param_bindings: tuple[InstanceParamBinding, ...]
    state_bindings: tuple[InstanceStateBinding, ...]
    shared_refs: tuple[str, ...]
    pending_slots: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "schema": EXPANSION_SCHEMA,
            "template_id": self.template_id,
            "n_stages": self.n_stages,
            "n_main_layers": self.n_main_layers,
            "param_specs": [s.to_dict() for s in self.param_specs],
            "param_bindings": [b.to_dict() for b in self.param_bindings],
            "state_bindings": [b.to_dict() for b in self.state_bindings],
            "shared_refs": list(self.shared_refs),
            "pending_slots": list(self.pending_slots),
        }


def _template_param_specs(values: ModelValues) -> tuple[TemplateParamSpec, ...]:
    """模板参数源清单（真实 shape/dtype/证据；数值来自 ModelValues）。

    形状/dtype 依据：mHC 参数 model.py:670-678（set_dtype fp32，N_mix=
    (2+N_hc)×N_hc，hc_dim=N_hc×H）；norm model.py:189-195（fp32）；
    Attention 基类参数 model.py:462-469（Linear 默认 dtype——Transformer
    default_dtype fp8，RMSNorm fp32；wo_a 显式 bf16）；gate model.py:562/
    567（TopK 分支；weight 存储 dtype = 参考入口 torch 默认 bf16——
    torch.empty 无显式 dtype + model.py:950 set_default_dtype，见
    ev_st_entry_default_dtype）；expert model.py:628-632（fp4/shared
    fp8）；main_proj model.py:832（first 专有，H×N_tgt→H）；last 专有
    norm/hc_head/markov/confidence model.py:836-841。"""
    n_mix = (2 + values.hc_mult) * values.hc_mult
    n_hch = values.hc_mult * values.dim
    e = values.n_routed_experts
    k = values.n_activated_experts
    moe_i = values.moe_inter_dim
    n_tgt = len(values.dspark_target_layer_ids)
    return (
        # --- 公共双 mHC（每实例独立，model.py:670-678）---
        TemplateParamSpec(
            "stage.hc_attn_fn_w", "hc_attn_fn", (n_mix, n_hch), "fp32",
            ("ev_st_hc_params",),
            "attention 侧 hc_pre 混合参数 [N_mix,N_hcH] fp32（set_dtype "
            "fp32 块）；消费点 stage.hc_pre_attn.mix.w"),
        TemplateParamSpec(
            "stage.hc_attn_scale_w", "hc_attn_scale", (3,), "fp32",
            ("ev_st_hc_params",),
            "attention 侧 hc_scale [3] fp32；消费点 "
            "stage.hc_pre_attn.sinkhorn.scale（pre/post/comb 三权重）"),
        TemplateParamSpec(
            "stage.hc_attn_base_w", "hc_attn_base", (n_mix,), "fp32",
            ("ev_st_hc_params",),
            "attention 侧 hc_base [N_mix] fp32；消费点 "
            "stage.hc_pre_attn.sinkhorn.base"),
        TemplateParamSpec(
            "stage.hc_ffn_fn_w", "hc_ffn_fn", (n_mix, n_hch), "fp32",
            ("ev_st_hc_params",),
            "ffn 侧 hc_pre 混合参数；消费点 stage.hc_pre_ffn.mix.w"),
        TemplateParamSpec(
            "stage.hc_ffn_scale_w", "hc_ffn_scale", (3,), "fp32",
            ("ev_st_hc_params",),
            "ffn 侧 hc_scale；消费点 stage.hc_pre_ffn.sinkhorn.scale"),
        TemplateParamSpec(
            "stage.hc_ffn_base_w", "hc_ffn_base", (n_mix,), "fp32",
            ("ev_st_hc_params",),
            "ffn 侧 hc_base；消费点 stage.hc_pre_ffn.sinkhorn.base"),
        # --- Block norm（model.py:716-717 → RMSNorm 189-195）---
        TemplateParamSpec(
            "stage.attn_norm_w", "attn_norm.weight", (values.dim,), "fp32",
            ("ev_st_norm_params", "ev_attn_norm"),
            "attn_norm RMSNorm weight [H] fp32（checkpoint 存 bf16，模块"
            " fp32——源码注释）；消费点 stage.attn_norm.w"),
        TemplateParamSpec(
            "stage.ffn_norm_w", "ffn_norm.weight", (values.dim,), "fp32",
            ("ev_st_norm_params", "ev_attn_norm"),
            "ffn_norm RMSNorm weight [H] fp32；消费点 stage.ffn_norm.w"),
        # --- Attention（基类参数，model.py:462-469）---
        TemplateParamSpec(
            "stage.wq_a_w", "attn.wq_a.weight",
            (values.q_lora_rank, values.dim), "fp8_e4m3fn",
            ("ev_ds_attn_init",),
            "低秩 Q 下投影 [Rq,H] fp8（Transformer default_dtype）；消费点 "
            "dspark.wq_a（decode-only，草稿 Q 链）"),
        TemplateParamSpec(
            "stage.q_norm_w", "attn.q_norm.weight", (values.q_lora_rank,),
            "fp32", ("ev_ds_attn_init", "ev_st_norm_params"),
            "q_norm RMSNorm weight [Rq] fp32；消费点 dspark.q_norm"),
        TemplateParamSpec(
            "stage.wq_b_w", "attn.wq_b.weight",
            (values.n_heads * values.head_dim, values.q_lora_rank),
            "fp8_e4m3fn", ("ev_ds_attn_init",),
            "Q 上投影 [Hq*D,Rq] fp8（ColumnParallelLinear，world_size=1 全宽）；"
            "消费点 dspark.wq_b"),
        TemplateParamSpec(
            "stage.wkv_w", "attn.wkv.weight", (values.head_dim, values.dim),
            "fp8_e4m3fn", ("ev_ds_attn_init", "ev_ds_main_kv", "ev_ds_draft_kv"),
            "共享单投影 KV [D,H] fp8；同一实例两次调用（main 759 / draft "
            "778）——消费点 dspark.wkv_main.w 两相位 + dspark.wkv_draft.w "
            "decode-only"),
        TemplateParamSpec(
            "stage.kv_norm_w", "attn.kv_norm.weight", (values.head_dim,),
            "fp32", ("ev_ds_attn_init", "ev_st_norm_params"),
            "kv_norm RMSNorm weight [D] fp32；同一实例两次调用（main/draft）"),
        TemplateParamSpec(
            "stage.wo_a_w", "attn.wo_a.weight",
            (values.o_groups * values.o_lora_rank,
             values.n_heads * values.head_dim // values.o_groups), "bf16",
            ("ev_ds_attn_init",),
            "O 分组低秩 [G*Ro,Hq*D/G] bf16（model.py:468 显式 dtype）；"
            "消费点 dspark.o_einsum（wo_a.weight.view(G,Ro,-1)）"),
        TemplateParamSpec(
            "stage.wo_b_w", "attn.wo_b.weight", (values.dim,
                                                 values.o_groups
                                                 * values.o_lora_rank),
            "fp8_e4m3fn", ("ev_ds_attn_init",),
            "O 输出投影 [H,G*Ro] fp8（RowParallelLinear）；消费点 "
            "dspark.wo_b / stage.hc_post_attn.combine.x 上游"),
        TemplateParamSpec(
            "stage.attn_sink_w", "attn.attn_sink", (values.n_heads,),
            "fp32", ("ev_ds_attn_init",),
            "attn_sink [Hq] fp32（model.py:462）；消费点 dspark.sparse_attn"
            ".sink（decode-only）"),
        # --- TopK MoE gate（model.py:562/567，hash=False 分支）---
        TemplateParamSpec(
            "stage.gate_weight_w", "ffn.gate.weight", (e, values.dim),
            "bf16", ("ev_st_gate_params", "ev_st_entry_default_dtype",
                     "ev_st_linear_dtype_mech", "ev_st_module_default_dtype"),
            "gate.weight [E,H] 存储 bf16（model.py:562 torch.empty 无显式 "
            "dtype → torch 默认 dtype；参考入口 model.py:950 "
            "set_default_dtype(bfloat16)；884 default_dtype 是模块变量供"
            "自定义 Linear 用，不能改变 torch.empty 默认 dtype；Gate."
            "forward 570 weight.float() 是 fp32 求值非存储）；消费点 "
            "stage.moe.gate.score.w"),
        TemplateParamSpec(
            "stage.gate_bias_w", "ffn.gate.bias", (e,), "fp32",
            ("ev_st_gate_params",),
            "gate.bias [E] fp32（TopK 分支，stage 层 43/44/45 ≥ "
            "n_hash_layers）；消费点 stage.moe.gate.bias.b（只进选择，"
            "不进权重 gather）"),
        # --- 专家（model.py:628-632；模板单专家身份 × E 实例在 checkpoint
        # 展开，图以 expert_template 表达模板、不编造 E 份独立记录）---
        TemplateParamSpec(
            "stage.expert_w1_w", "ffn.experts.*.w1.weight",
            (moe_i, values.dim), "fp4_e2m1fn_x2", ("ev_st_expert_dtype",
                                                  "ev_expert"),
            "路由专家 SwiGLU w1 [D_in,H]（checkpoint [D_in,H//2] + scale，"
            "expert_dtype=fp4）；模板身份，E=256 份实例；消费点 "
            "stage.moe.expert_template（x[idx] 子集）"),
        TemplateParamSpec(
            "stage.expert_w3_w", "ffn.experts.*.w3.weight",
            (moe_i, values.dim), "fp4_e2m1fn_x2", ("ev_st_expert_dtype",
                                                  "ev_expert"),
            "路由专家 SwiGLU up w3；同 expert_w1 模板身份"),
        TemplateParamSpec(
            "stage.expert_w2_w", "ffn.experts.*.w2.weight",
            (values.dim, moe_i), "fp4_e2m1fn_x2", ("ev_st_expert_dtype",
                                                  "ev_expert"),
            "路由专家 SwiGLU down w2 [H,D_in]；同 expert_w1 模板身份"),
        TemplateParamSpec(
            "stage.shared_w1_w", "ffn.shared_experts.w1.weight",
            (moe_i, values.dim), "fp8_e4m3fn", ("ev_st_expert_dtype",
                                               "ev_moe_shared"),
            "shared expert w1（default dtype fp8，无 fp4 分支）；消费点 "
            "stage.moe.shared_experts"),
        TemplateParamSpec(
            "stage.shared_w2_w", "ffn.shared_experts.w2.weight",
            (values.dim, moe_i), "fp8_e4m3fn", ("ev_st_expert_dtype",
                                               "ev_moe_shared"),
            "shared expert w2"),
        TemplateParamSpec(
            "stage.shared_w3_w", "ffn.shared_experts.w3.weight",
            (moe_i, values.dim), "fp8_e4m3fn", ("ev_st_expert_dtype",
                                               "ev_moe_shared"),
            "shared expert w3"),
        # --- first 专有（stage0，model.py:830-833）---
        TemplateParamSpec(
            "stage.main_proj_w", "main_proj.weight",
            (values.dim, values.dim * n_tgt), "fp8_e4m3fn",
            ("ev_st_main_proj",),
            "first 专有 main_proj [H,H*N_tgt]（=H×3 readout concat 输入宽，"
            "EXPR_MAIN_PROJ_IN）；只在 stage0 实例展开"),
        TemplateParamSpec(
            "stage.main_norm_w", "main_norm.weight", (values.dim,), "fp32",
            ("ev_st_main_proj", "ev_st_norm_params"),
            "first 专有 main_norm RMSNorm weight [H] fp32；只在 stage0"),
        # --- last 专有（stage n-1，model.py:836-841）---
        TemplateParamSpec(
            "stage.norm_w", "norm.weight", (values.dim,), "fp32",
            ("ev_st_stage_role", "ev_st_norm_params"),
            "last 专有 norm RMSNorm weight [H] fp32（forward_head 用）；"
            "只在 last stage"),
        TemplateParamSpec(
            "stage.hc_head_fn_w", "hc_head_fn", (values.hc_mult, n_hch),
            "fp32", ("ev_st_hc_head_stage2",),
            "last 专有 hc_head_fn [N_hc,N_hcH] fp32（set_dtype 块，"
            "model.py:838-841）；forward_head 内部图 pending"),
        TemplateParamSpec(
            "stage.hc_head_base_w", "hc_head_base", (values.hc_mult,),
            "fp32", ("ev_st_hc_head_stage2",),
            "last 专有 hc_head_base [N_hc] fp32"),
        TemplateParamSpec(
            "stage.hc_head_scale_w", "hc_head_scale", (1,), "fp32",
            ("ev_st_hc_head_stage2",),
            "last 专有 hc_head_scale [1] fp32"),
        TemplateParamSpec(
            "stage.markov_w1_w", "markov_head.markov_w1.weight",
            (values.vocab_size, values.dspark_markov_rank), "bf16",
            ("ev_st_markov_head",),
            "last 专有 markov_w1 ParallelEmbedding [V,Rm]；头内部图 pending"),
        TemplateParamSpec(
            "stage.markov_w2_w", "markov_head.markov_w2.weight",
            (values.vocab_size, values.dspark_markov_rank), "fp32",
            ("ev_st_markov_head",),
            "last 专有 markov_w2 ParallelHead [V,Rm] fp32（ParallelHead "
            "weight fp32，model.py:722-724）；头内部图 pending"),
        TemplateParamSpec(
            "stage.confidence_proj_w", "confidence_head.proj.weight",
            (1, values.dim + values.dspark_markov_rank), "fp32",
            ("ev_st_markov_head",),
            "last 专有 confidence proj [1,H+Rm] fp32（dtype=torch.float32，"
            "model.py:809）；头内部图 pending"),
    )


_FIRST_ONLY_SPECS = frozenset({"stage.main_proj_w", "stage.main_norm_w"})
_LAST_ONLY_SPECS = frozenset({
    "stage.norm_w", "stage.hc_head_fn_w", "stage.hc_head_base_w",
    "stage.hc_head_scale_w", "stage.markov_w1_w", "stage.markov_w2_w",
    "stage.confidence_proj_w",
})
# 专家模板 spec 的 checkpoint 前缀在 ffn.experts.*（模板通配），实例路径
# 保持模板通配形式（不编造 256 份独立路径清单——展开绑定表达模板→实例
# 前缀关系，E 份权重由 checkpoint 布局承载）。
_WILDCARD_SPECS = frozenset({
    "stage.expert_w1_w", "stage.expert_w3_w", "stage.expert_w2_w",
})

# 模板 cache state 可信事实（B2 补修 02）：锚定 B1 单一实现（deepseek_v4_
# dspark._states：state "dspark:win_cache"、transition "tr_dspark_cache"、
# 默认两版本 v_hist→v_after）与 stage 图真实张量。构建期
# _verify_state_facts 逐项对照当前 stage 图核对，防常量与真实图漂移；
# 读回核对以常量为期望，不用被检文档字段拼。cache_tensor 字段语义 =
# 模板 stage 图内张量的 name（stable ID = "tensor:" + name），非显示名、
# 非 stable ID 本身。
TEMPLATE_STATE_ID = "dspark:win_cache"
TEMPLATE_STATE_TRANSITION = "tr_dspark_cache"
TEMPLATE_STATE_VERSIONS = ("v_hist", "v_after")
TEMPLATE_CACHE_TENSOR_NAME = "stage:win_cache"


def build_template_expansion(values: ModelValues,
                             template_id: str = STAGE_TEMPLATE_ID
                             ) -> StageExpansion:
    """参数/状态展开绑定表（纯派生，数值来自 ModelValues，不改固定输入）。

    - 公共参数（双 mHC/norm/Attention/MoE gate/expert/shared）：每实例
      独立展开 mtp.{i}.{slot}；
    - first/last 角色专有参数：只展开到对应实例（first=stage0、
      last=stage n-1；single stage 并集）；
    - cache 状态：模板 dspark:win_cache → dspark:win_cache:mtp.{i} 独立
      身份（版本对 v_hist→v_after、transition tr_dspark_cache、layer 43+i）；
    - shared embed/head：引用 Transformer 身份（model.py:903-904 同一
      对象），不复制三份路径。"""
    n = values.n_mtp_layers
    n_main = values.n_layers
    if n < 1 or n_main < 1:
        raise ExtractionError(
            f"n_mtp_layers/n_layers 必须 ≥1，得到 {n}/{n_main}")
    specs = _template_param_specs(values)
    by_id = {s.spec_id: s for s in specs}

    param_bindings: list[InstanceParamBinding] = []
    for i in range(n):
        single = (i == 0 == n - 1)
        for s in specs:
            if s.spec_id in _FIRST_ONLY_SPECS and not (i == 0 or single):
                continue
            if s.spec_id in _LAST_ONLY_SPECS and not (i == n - 1 or single):
                continue
            if s.param_slot is None:
                raise ExtractionError(
                    f"参数 spec {s.spec_id} 缺 param_slot（展开表要求全部 "
                    "参数槽可定位）")
            slot = s.param_slot
            if s.spec_id in _WILDCARD_SPECS:
                # ffn.experts.*.wN.weight → 保留 wN 层：mtp.{i}.ffn.
                # experts.*.wN.weight（E 份实例由 checkpoint 布局承载，
                # 绑定表记录模板→实例前缀关系）
                path = f"mtp.{i}.{slot}"
            else:
                path = f"mtp.{i}.{slot}"
            param_bindings.append(InstanceParamBinding(
                spec_id=s.spec_id, stage_id=i, instance_path=path,
                shape=s.shape, dtype=s.dtype))

    state_bindings = tuple(
        InstanceStateBinding(
            stage_id=i,
            template_state_id=TEMPLATE_STATE_ID,
            state_id=f"{TEMPLATE_STATE_ID}:mtp.{i}",
            cache_tensor=TEMPLATE_CACHE_TENSOR_NAME,
            version_labels=TEMPLATE_STATE_VERSIONS,
            transition_id=TEMPLATE_STATE_TRANSITION,
            layer_id=n_main + i,
        )
        for i in range(n)
    )

    pending_slots = (
        "first 入口完整图 pending（main_proj/main_norm/forward_embed "
        "展开只到参数路径与消费点，内部节点级子图未提取）",
        "last 出口完整图 pending（norm/hc_head/markov_head/"
        "confidence_head 同上）",
    )
    return StageExpansion(
        n_stages=n, n_main_layers=n_main, template_id=template_id,
        param_specs=specs, param_bindings=tuple(param_bindings),
        state_bindings=state_bindings,
        shared_refs=("shared.embed", "shared.head"),
        pending_slots=pending_slots,
    )


def _verify_expansion(exp: StageExpansion, values: ModelValues) -> None:
    """展开表一致性核对（读回必检，B2 补修 02 收紧）：期望来源是固定
    可信 ModelValues（固定输入 config 派生，调用方经 load_model_values
    取得）经 _template_param_specs 派生的完整规范参数目录——不是被检
    文档自身，spec 与 bindings 共同篡改（整槽删除、shape/dtype 两边
    一致地改）不通过。核对项：
    - 模板身份；n_stages/n_main_layers 与固定 config 一致；
    - param_specs 集合恰等于可信目录（无缺失/无多余/无重复），每条
      spec 的 param_slot/shape/dtype 逐字段与可信目录一致；
    - param_bindings 覆盖完整（公共槽每实例必有、角色槽角色规则），
      instance_path 恰为 mtp.{stage_id}.{param_slot}（含源码模板通配
      槽，完整路径相等，非 startswith 前缀），shape/dtype 与可信 spec
      一致；
    - state_bindings：template_state_id/版本对/transition/cache 张量
      name 锚定 B1 单一实现常量（TEMPLATE_STATE_*，构建期经
      _verify_state_facts 与真实 stage 图核对），state_id 恰为
      {template}:mtp.{stage_id}（不从待检 template_state_id 拼），每
      实例唯一 namespace/层号。
    任何不符具名拒绝——包括实例间复制/交换身份、以被检文档自身拼
    “完整”。"""
    if exp.template_id != STAGE_TEMPLATE_ID:
        raise ExtractionError(
            f"expansion template_id 不符: {exp.template_id!r}")
    if exp.n_stages != values.n_mtp_layers or exp.n_main_layers != values.n_layers:
        raise ExtractionError(
            f"expansion n_stages/n_main_layers（{exp.n_stages}/"
            f"{exp.n_main_layers}）与固定 config（{values.n_mtp_layers}/"
            f"{values.n_layers}）不一致——期望目录以固定来源为准")
    expected_specs = _template_param_specs(values)
    expected_by_id = {s.spec_id: s for s in expected_specs}
    spec_ids = {s.spec_id for s in exp.param_specs}
    if len(spec_ids) != len(exp.param_specs):
        raise ExtractionError("expansion param_specs 有重复 spec_id")
    missing_ids = sorted(set(expected_by_id) - spec_ids)
    if missing_ids:
        raise ExtractionError(
            f"param_specs 缺可信目录必需槽: {missing_ids}——spec 及全部 "
            "bindings 共同删除的整槽篡改具名拒绝（期望目录来自固定来源，"
            "不以被检文档自身推导“完整”）")
    extra_ids = sorted(spec_ids - set(expected_by_id))
    if extra_ids:
        raise ExtractionError(
            f"param_specs 含可信目录之外的槽: {extra_ids}")
    for s in exp.param_specs:
        e = expected_by_id[s.spec_id]
        if s.param_slot != e.param_slot:
            raise ExtractionError(
                f"param spec {s.spec_id} param_slot {s.param_slot!r} 与可信"
                f"目录 {e.param_slot!r} 不一致")
        if s.shape != e.shape:
            raise ExtractionError(
                f"param spec {s.spec_id} shape {s.shape} 与可信目录 "
                f"{e.shape} 不一致——spec 与 bindings 共同篡改 shape 具名"
                "拒绝")
        if s.dtype != e.dtype:
            raise ExtractionError(
                f"param spec {s.spec_id} dtype {s.dtype!r} 与可信目录 "
                f"{e.dtype!r} 不一致——spec 与 bindings 共同篡改 dtype 具名"
                "拒绝")
    by_id = {s.spec_id: s for s in exp.param_specs}

    seen_paths: set[str] = set()
    seen_state: set[str] = set()
    per_stage_specs: dict[int, set[str]] = {i: set()
                                            for i in range(exp.n_stages)}
    for b in exp.param_bindings:
        if b.spec_id not in spec_ids:
            raise ExtractionError(
                f"param_binding 引用未知 spec_id: {b.spec_id!r}")
        if not (0 <= b.stage_id < exp.n_stages):
            raise ExtractionError(
                f"param_binding stage_id={b.stage_id} 越界（n_stages="
                f"{exp.n_stages}）")
        spec = by_id[b.spec_id]
        expect_path = f"mtp.{b.stage_id}.{spec.param_slot}"
        if b.instance_path != expect_path:
            raise ExtractionError(
                f"param_binding {b.instance_path!r} 不恰为可信映射 "
                f"{expect_path!r}（stage_id={b.stage_id} 的 spec 槽 "
                f"{spec.param_slot!r}）——合法前缀但槽位不对应/实例间复制"
                "交换身份具名拒绝")
        if b.instance_path in seen_paths:
            raise ExtractionError(
                f"param_binding instance_path 重复: {b.instance_path!r}")
        seen_paths.add(b.instance_path)
        if b.shape != spec.shape:
            raise ExtractionError(
                f"param_binding {b.instance_path} shape {b.shape} 与模板 "
                f"spec {spec.shape} 不一致")
        if b.dtype != spec.dtype:
            raise ExtractionError(
                f"param_binding {b.instance_path} dtype {b.dtype} 与模板 "
                f"spec {spec.dtype} 不一致")
        per_stage_specs[b.stage_id].add(b.spec_id)

    # 角色专有槽覆盖规则：first 槽必须且只能在 first 实例展开；last 同理。
    for i in range(exp.n_stages):
        have = per_stage_specs[i]
        first_ok = (i == 0) or exp.n_stages == 1
        last_ok = (i == exp.n_stages - 1) or exp.n_stages == 1
        for s in _FIRST_ONLY_SPECS:
            if s in spec_ids and (s in have) != first_ok:
                raise ExtractionError(
                    f"first 专有槽 {s} 在 stage {i} 的展开违反角色规则")
        for s in _LAST_ONLY_SPECS:
            if s in spec_ids and (s in have) != last_ok:
                raise ExtractionError(
                    f"last 专有槽 {s} 在 stage {i} 的展开违反角色规则")
        # 公共槽必须每实例都有
        for s in spec_ids:
            if (s not in _FIRST_ONLY_SPECS and s not in _LAST_ONLY_SPECS
                    and s not in have):
                raise ExtractionError(
                    f"公共参数槽 {s} 缺 stage {i} 实例展开")

    for sb in exp.state_bindings:
        if not (0 <= sb.stage_id < exp.n_stages):
            raise ExtractionError(
                f"state_binding stage_id={sb.stage_id} 越界")
        if sb.template_state_id != TEMPLATE_STATE_ID:
            raise ExtractionError(
                f"state_binding template_state_id {sb.template_state_id!r} "
                f"与模板 state {TEMPLATE_STATE_ID!r} 不符——合法 state 以"
                "固定源码模板（B1 _states 实现）为准，不从待检字段拼期望")
        if sb.cache_tensor != TEMPLATE_CACHE_TENSOR_NAME:
            raise ExtractionError(
                f"state_binding cache_tensor {sb.cache_tensor!r} 与模板 "
                f"cache 张量 {TEMPLATE_CACHE_TENSOR_NAME!r} 不符")
        expect_state = f"{TEMPLATE_STATE_ID}:mtp.{sb.stage_id}"
        if sb.state_id != expect_state:
            raise ExtractionError(
                f"state_binding {sb.state_id!r} 与实例序号不对应（期望 "
                f"{expect_state!r}）——实例间复制/交换身份具名拒绝")
        if sb.state_id in seen_state:
            raise ExtractionError(f"state_id 重复: {sb.state_id!r}")
        seen_state.add(sb.state_id)
        if sb.layer_id != exp.n_main_layers + sb.stage_id:
            raise ExtractionError(
                f"state_binding layer_id={sb.layer_id} 与 n_main_layers+"
                f"stage_id（{exp.n_main_layers + sb.stage_id}）矛盾")
        if sb.version_labels != TEMPLATE_STATE_VERSIONS:
            raise ExtractionError(
                f"state_binding version_labels {sb.version_labels} 与模板"
                f"版本对 {TEMPLATE_STATE_VERSIONS} 不一致")
        if sb.transition_id != TEMPLATE_STATE_TRANSITION:
            raise ExtractionError(
                f"state_binding transition_id {sb.transition_id!r} 与模板"
                f" transition {TEMPLATE_STATE_TRANSITION!r} 不一致")
    if len(seen_state) != exp.n_stages:
        raise ExtractionError("state_binding 覆盖不完整")


def serialize_expansion(exp: StageExpansion) -> str:
    """展开表公开规范序列化（规范化 JSON，重复构建逐字节一致）。"""
    return json.dumps(exp.to_dict(), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def deserialize_expansion(text: str, values: ModelValues | None = None,
                          workspace_root: Path | None = None) -> StageExpansion:
    """展开表读回：schema/template_id 显式核对 + 重建后全量一致性核对
    （_verify_expansion——期望参数目录/state 事实来自固定可信 ModelValues，
    不以被检 JSON 充当可信基线；复制身份/整槽删除/spec+bindings 共同
    篡改 shape/dtype/假 state 具名拒绝）。values 缺省从固定工作区加载
    （load_model_values，固定输入只读）。"""
    root = Path(workspace_root) if workspace_root is not None \
        else find_workspace_root()
    values = values if values is not None else load_model_values(root)
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"expansion JSON 不合法: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != EXPANSION_SCHEMA:
        raise ExtractionError(
            f"expansion schema 不符，期望 {EXPANSION_SCHEMA}")
    for key in ("template_id", "n_stages", "n_main_layers", "param_specs",
                "param_bindings", "state_bindings", "shared_refs",
                "pending_slots"):
        if key not in doc:
            raise ExtractionError(f"expansion 缺字段: {key}")
    exp = StageExpansion(
        n_stages=doc["n_stages"], n_main_layers=doc["n_main_layers"],
        template_id=doc["template_id"],
        param_specs=tuple(TemplateParamSpec.from_dict(s)
                          for s in doc["param_specs"]),
        param_bindings=tuple(InstanceParamBinding.from_dict(b)
                             for b in doc["param_bindings"]),
        state_bindings=tuple(InstanceStateBinding.from_dict(b)
                             for b in doc["state_bindings"]),
        shared_refs=tuple(doc["shared_refs"]),
        pending_slots=tuple(doc["pending_slots"]),
    )
    _verify_expansion(exp, values)
    return exp


def _verify_binding_consistency(bindings: tuple[StageBinding, ...],
                                n_main_layers: int | None = None) -> None:
    """实例身份一致性（读回必检，ChainMap 读回复用）：参数路径/状态 ID
    与序号一一对应、actual_layer_id = n_main_layers + 序号、全表唯一。
    实例间复制身份（如第 1 实例携带第 0 实例 param_path/state_id）具名
    拒绝，不允许静默接受。n_main_layers 缺省时从首条绑定派生
    （actual_layer_id − stage_id，即 constructor 编号基准）。"""
    if n_main_layers is None:
        if not bindings:
            raise ExtractionError("bindings 为空，无法核对实例身份")
        n_main_layers = (bindings[0].actual_layer_id
                         - bindings[0].stage_id)
    seen_params: set[str] = set()
    seen_states: set[str] = set()
    for i, b in enumerate(bindings):
        expect_param = f"mtp.{i}"
        if b.param_path != expect_param:
            raise ExtractionError(
                f"bindings[{i}].param_path={b.param_path!r} 与实例序号不"
                f"对应（期望 {expect_param!r}）——实例间复制/交换身份具名拒绝")
        if b.param_path in seen_params:
            raise ExtractionError(f"param_path 重复: {b.param_path!r}")
        seen_params.add(b.param_path)
        expect_state = f"dspark:win_cache:mtp.{i}"
        if b.state_id != expect_state:
            raise ExtractionError(
                f"bindings[{i}].state_id={b.state_id!r} 与实例序号不对应"
                f"（期望 {expect_state!r}）——实例间复制/交换身份具名拒绝")
        if b.state_id in seen_states:
            raise ExtractionError(f"state_id 重复: {b.state_id!r}")
        seen_states.add(b.state_id)
        expect_layer = n_main_layers + i
        if b.actual_layer_id != expect_layer:
            raise ExtractionError(
                f"bindings[{i}].actual_layer_id={b.actual_layer_id} 与"
                f" n_main_layers+序号（{expect_layer}）矛盾")
        if b.template_id != STAGE_TEMPLATE_ID:
            raise ExtractionError(
                f"bindings[{i}].template_id 不符: {b.template_id!r}")


def serialize_bindings(bindings: tuple[StageBinding, ...]) -> str:
    """绑定记录公开规范序列化（规范化 JSON，重复构建逐字节一致）。"""
    doc = {
        "schema": BINDING_SCHEMA,
        "template_id": STAGE_TEMPLATE_ID,
        "n_stages": len(bindings),
        "bindings": [b.to_dict() for b in bindings],
    }
    return json.dumps(doc, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def deserialize_bindings(text: str) -> tuple[StageBinding, ...]:
    """绑定记录读回：schema/template_id/n_stages 显式核对；角色一致性
    （首/末布尔与序号关系）重新验证；结构不合法具名抛 ExtractionError。"""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"bindings JSON 不合法: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != BINDING_SCHEMA:
        raise ExtractionError(
            f"bindings schema 不符，期望 {BINDING_SCHEMA}，"
            f"得到 {doc.get('schema') if isinstance(doc, dict) else doc!r}")
    if doc.get("template_id") != STAGE_TEMPLATE_ID:
        raise ExtractionError("bindings template_id 不符")
    raw = doc.get("bindings")
    if not isinstance(raw, list) or not raw:
        raise ExtractionError("bindings 必须是非空 list")
    if doc.get("n_stages") != len(raw):
        raise ExtractionError("n_stages 与 bindings 长度不一致")
    out = tuple(StageBinding.from_dict(b) for b in raw)
    for i, b in enumerate(out):
        if b.stage_id != i:
            raise ExtractionError(f"bindings[{i}].stage_id 必须等于序号 {i}")
        if b.is_first != (i == 0):
            raise ExtractionError(f"bindings[{i}].is_first 与序号矛盾")
        if b.is_last != (i == len(out) - 1):
            raise ExtractionError(f"bindings[{i}].is_last 与序号矛盾")
    _verify_binding_consistency(out, n_main_layers=out[0].actual_layer_id
                                - out[0].stage_id)
    return out


@dataclass(frozen=True)
class ChainMap:
    """三 stage 链路边界映射（model.py:929-935 拓扑事实 + readout binding
    + compress_ratios/nextn 消费说明）。

    - stage_bindings：实例绑定记录（derive_stage_bindings 输出）；
    - chain_edges：h 串接与 main_x 分发的边列表（src/dst 用 param_path
      定位实例，每条边显式 phase/guard 字段，note 锚定证据）；
    - readout_binding：40/41/42→mean(hc)→有序 concat→stage0 main_proj/
      main_norm 来源关系（有证据的 binding/consumer 记录，不把 readout
      错误对应到各 stage——三个 readout 全部汇入 stage0 唯一 main_proj）；
    - ratios_consumers：compress_ratios 46 项布局的消费说明（前 43 主干/
      后 3 draft，尾部全 0）；
    - nextn_note：root config num_nextn_predict_layers=1 与 run config
      n_mtp_layers=3 的不同来源/消费者说明。
    """

    stage_bindings: tuple[StageBinding, ...]
    chain_edges: tuple[dict, ...]
    readout_binding: dict
    ratios_consumers: dict
    nextn_note: dict

    def to_dict(self) -> dict:
        return {
            "schema": CHAIN_SCHEMA,
            "template_id": STAGE_TEMPLATE_ID,
            "stage_bindings": [b.to_dict() for b in self.stage_bindings],
            "chain_edges": list(self.chain_edges),
            "readout_binding": self.readout_binding,
            "ratios_consumers": self.ratios_consumers,
            "nextn_note": self.nextn_note,
        }


def build_chain_map(
    workspace_root: Path | None = None,
    *,
    values: ModelValues | None = None,
) -> ChainMap:
    """三 stage 链路图/边界映射（model.py:929-935 + 917-925 + 830-833）。

    纯派生（读 ModelValues 数值，不修改固定输入）；边全部带证据锚定。"""
    root = Path(workspace_root) if workspace_root is not None else find_workspace_root()
    values = values or load_model_values(root)
    targets = list(values.dspark_target_layer_ids)  # (40, 41, 42)
    n_main = values.n_layers                        # 43
    if len(targets) < 1:
        raise ExtractionError("dspark_target_layer_ids 为空（model.py:831 "
                              "assert 长度 >0）")
    bindings = derive_stage_bindings(values.n_mtp_layers, n_main)

    # 链路边（forward_spec model.py:929-935 拓扑；note 锚定证据）。
    # 每条边显式携带 phase（decode/prefill/both）与 guard（可求值谓词或
    # null=两相位无条件）——条件不靠 note 暗示。forward_spec:931-934
    # prefill 也执行 for layer 并传递 h（各 stage prefill 分支 passthrough
    # 返回原 h，不算草稿双 mHC 计算）；草稿计算消费只在 decode。
    chain_edges = (
        {
            "edge_id": "chain.embed_h",
            "src": "mtp.0.forward_embed", "src_port": "h",
            "dst": "mtp.0", "dst_port": "h_in",
            "kind": "data",
            "phase": "both",
            "guard": None,
            "note": "草稿 h 来源：stage0.forward_embed 产出的 HC 展开流"
                    "（shared.embed 查表→unsqueeze→repeat，model.py:857-861）"
                    "——不是原始 transformer.embed.y 直接当 HC 流；embed 属"
                    " first 入口内部图（pending），此处只记录链路边界",
            "evidence": ["ev_st_forward_spec", "ev_st_forward_embed"],
        },
        {
            "edge_id": "chain.main_x_stage0",
            "src": "mtp.0.forward_embed", "src_port": "main_x",
            "dst": "mtp.0", "dst_port": "main_x",
            "kind": "data",
            "phase": "both",
            "guard": None,
            "note": "main_x 生产点：stage0 main_norm(main_proj(main_hidden))"
                    "（model.py:854）",
            "evidence": ["ev_st_forward_embed", "ev_st_main_proj"],
        },
        {
            "edge_id": "chain.main_x_stage1",
            "src": "mtp.0.forward_embed", "src_port": "main_x",
            "dst": "mtp.1", "dst_port": "main_x",
            "kind": "data",
            "phase": "both",
            "guard": None,
            "note": "同一 main_x 张量传给 stage1（第三实参逐层透传，"
                    "非各 stage 独立 readout）",
            "evidence": ["ev_st_forward_spec"],
        },
        {
            "edge_id": "chain.main_x_stage2",
            "src": "mtp.0.forward_embed", "src_port": "main_x",
            "dst": "mtp.2", "dst_port": "main_x",
            "kind": "data",
            "phase": "both",
            "guard": None,
            "note": "同一 main_x 张量传给 stage2",
            "evidence": ["ev_st_forward_spec"],
        },
        {
            "edge_id": "chain.h_0_1",
            "src": "mtp.0", "src_port": "h_out",
            "dst": "mtp.1", "dst_port": "h_in",
            "kind": "data",
            "phase": "decode",
            "guard": "start_pos > 0",
            "note": "decode 草稿计算串接 stage0→1（for layer in self.mtp: "
                    "h = layer(h, start_pos, input_ids, main_x)）——h 消费"
                    "进入 stage1 双 mHC（Block.forward 696 hc_pre）",
            "evidence": ["ev_st_forward_spec", "ev_ds_block_prefill"],
        },
        {
            "edge_id": "chain.h_1_2",
            "src": "mtp.1", "src_port": "h_out",
            "dst": "mtp.2", "dst_port": "h_in",
            "kind": "data",
            "phase": "decode",
            "guard": "start_pos > 0",
            "note": "decode 草稿计算串接 stage1→2",
            "evidence": ["ev_st_forward_spec", "ev_ds_block_prefill"],
        },
        {
            "edge_id": "chain.h_head_pending",
            "src": "mtp.2", "src_port": "h_out",
            "dst": "mtp.2.forward_head", "dst_port": "x",
            "kind": "data",
            "phase": "decode",
            "guard": "start_pos > 0",
            "note": "末 stage h 输出留待 forward_head（仅 decode，"
                    "forward_spec 934 相位分支）；头内部图 pending",
            "evidence": ["ev_st_forward_spec", "ev_st_forward_head"],
        },
        {
            "edge_id": "chain.prefill_h_passthrough",
            "src": "mtp.{i}", "src_port": "h_in",
            "dst": "mtp.{i}", "dst_port": "h_out",
            "kind": "data",
            "phase": "prefill",
            "guard": "start_pos == 0",
            "note": "prefill（start_pos==0）：forward_spec:931-934 for 循环"
                    "仍执行并传递 h（h = layer(h, start_pos, input_ids, "
                    "main_x)），但各 stage DSparkBlock.forward 845-849 走 "
                    "prefill 分支——只调 attn 构建 win_cache，attn prefill "
                    "分支 769 return x 原样，h passthrough 返回（各 stage "
                    "返回相同 h），不算草稿双 mHC 计算；三 cache 构建、"
                    "无 head（934 return）",
            "evidence": ["ev_st_forward_spec", "ev_ds_block_prefill"],
        },
    )

    # readout 汇聚 binding：三个主干 readout（40/41/42）→ 有序 concat →
    # stage0 main_proj/main_norm（唯一生产点）。不把 readout 对应到 stage。
    readout_binding = {
        "consumers": [
            {"layer": lid, "op": "h.mean(dim=2)", "out": "[B,T,H]",
             "order": k}
            for k, lid in enumerate(targets)
        ],
        "concat": {"op": "torch.cat(main_hiddens, dim=-1)",
                   "out": "[B,T,H*N_tgt]",
                   "order": "dspark_target_layer_ids 顺序 "
                            f"{list(targets)}"},
        "consumer_binding": {
            "param_path": "mtp.0",
            "module": "main_proj",
            "in_width_expr": EXPR_MAIN_PROJ_IN,
            "then": "main_norm",
            "out": "main_x [B,T,H] 共享分发三 stage",
        },
        "note": "三个 readout 全部汇入 stage0 唯一 main_proj/main_norm；"
                "不把各 readout 对应到各 stage（无 readout→stage1/2 直接边）",
        "evidence": ["ev_st_readout", "ev_st_main_proj",
                     "ev_st_forward_embed", "ev_st_forward_spec"],
    }

    # compress_ratios 消费说明
    ratios = list(values.compress_ratios)
    ratios_consumers = {
        "total_len": len(ratios),
        "main_range": [0, n_main - 1],
        "main_consumers": "Block 层 0-42（model.py:459 "
                          "compress_ratios[layer_id]，含 hash 层 0-2 / "
                          "CSA r=4 / HCA r=128）",
        "draft_range": [n_main, len(ratios) - 1],
        "draft_consumers": "DSparkBlock 层 43/44/45（同一取址式，stage 层 "
                           "compress_ratio==0——model.py:753 assert；尾部 3 项"
                           "全 0）",
        "draft_tail_values": ratios[n_main:],
        "evidence": ["ev_st_ratios_layout", "ev_ds_capacity"],
    }

    nextn_note = {
        "root_config_field": "num_nextn_predict_layers",
        "root_config_value": 1,
        "root_config_line": 31,
        "consumer": "checkpoint nextn.* 单 MTP 段布局（v3 MTPLoader 惯例，"
                    "非推理图实例数）",
        "run_config_field": "n_mtp_layers",
        "run_config_value": values.n_mtp_layers,
        "consumer_note": "inference model.py DSparkBlock 实例数（model.py:901 "
                         "range(args.n_mtp_layers)）→ 本模块实例绑定",
        "evidence": ["ev_st_nextn", "ev_st_layer_ids"],
    }

    return ChainMap(stage_bindings=bindings, chain_edges=chain_edges,
                    readout_binding=readout_binding,
                    ratios_consumers=ratios_consumers, nextn_note=nextn_note)


def chain_map_to_json(cm: ChainMap) -> str:
    """链路图公开规范序列化（规范化 JSON，重复构建逐字节一致）。"""
    return json.dumps(cm.to_dict(), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def chain_map_from_json(text: str) -> ChainMap:
    """链路图读回：schema/template_id 显式核对；结构不合法具名抛
    ExtractionError。"""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"chain map JSON 不合法: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != CHAIN_SCHEMA:
        raise ExtractionError(f"chain map schema 不符，期望 {CHAIN_SCHEMA}")
    if doc.get("template_id") != STAGE_TEMPLATE_ID:
        raise ExtractionError("chain map template_id 不符")
    for key in ("stage_bindings", "chain_edges", "readout_binding",
                "ratios_consumers", "nextn_note"):
        if key not in doc:
            raise ExtractionError(f"chain map 缺字段: {key}")
    bindings = tuple(StageBinding.from_dict(b) for b in doc["stage_bindings"])
    _verify_binding_consistency(bindings)
    return ChainMap(        stage_bindings=bindings,
        chain_edges=tuple(doc["chain_edges"]),
        readout_binding=doc["readout_binding"],
        ratios_consumers=doc["ratios_consumers"],
        nextn_note=doc["nextn_note"],
    )


# ---------------------------------------------------------------------------
# bundle 与入口
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DSparkStageBundle:
    """B2 交付包：公共 stage 图 + 证据注册表 + 模型数值 + 坐标表达式记录
    + Attention 子图边界映射 + 参数/状态展开绑定表。"""

    graph: SemanticGraph
    registry: EvidenceRegistry
    values: ModelValues
    coordinate_expressions: tuple[CoordinateExpression, ...]
    boundary_mappings: tuple[FoldBoundaryMapping, ...]
    chain: ChainMap
    expansion: StageExpansion
    attention_template_id: str = ATTENTION_TEMPLATE_ID


def _verify_fold_members(graph: SemanticGraph,
                         fold_members: tuple[str, ...],
                         mappings: tuple[FoldBoundaryMapping, ...]) -> None:
    """fold 成员/边界映射当前图真实性核对（构建期与测试共用）：fold 每个
    成员必须是本图真实节点；每条边界映射的内部节点必须是 fold 成员且
    端口方向/存在正确——missing 直接具名拒绝，不允许 continue 跳过。"""
    node_map = {n.stable_id: n for n in graph.nodes}
    for m in fold_members:
        if m not in node_map:
            raise ExtractionError(
                f"fold 组成员不属于当前 stage 图: {m!r}")
    for m in mappings:
        n = node_map.get(m.internal_node_id)
        if n is None:
            raise ExtractionError(
                f"边界映射内部节点不在当前图: {m.internal_node_id!r}")
        if m.internal_node_id not in fold_members:
            raise ExtractionError(
                f"边界映射内部节点不是 fold 组成员: {m.internal_node_id!r}")
        ports = n.output_ports if m.direction == "output" else n.input_ports
        if not any(p.name == m.internal_port for p in ports):
            raise ExtractionError(
                f"边界映射端口不存在: {m.internal_node_id}.{m.internal_port}")


def _verify_state_facts(graph: SemanticGraph) -> None:
    """模板 cache state 常量与当前 stage 图逐项核对（B2 补修 02，构建期）：
    state 版本（dspark:win_cache 的 v_hist/v_after）、transition
    （tr_dspark_cache 属该 state）、cache 张量（tensor:stage:win_cache，
    cache_tensor 字段语义 = name，stable ID = "tensor:"+name）、state 边
    引用 transition。展开表 state 绑定读回核对以这些常量为期望；常量若
    与真实图漂移，构建即具名拒绝——合法 state/version/transition 映射
    与真实 stage 图对应，不是拼字符串。"""
    versions = {(v.state_id, v.version_label) for v in graph.state_versions}
    for label in TEMPLATE_STATE_VERSIONS:
        if (TEMPLATE_STATE_ID, label) not in versions:
            raise ExtractionError(
                f"stage 图缺模板 state 版本 ({TEMPLATE_STATE_ID!r}, "
                f"{label!r})——TEMPLATE_STATE_* 常量与真实图不一致，拒绝"
                "以拼字符串 state 充当合法映射")
    trans = {t.stable_id: t.state_id for t in graph.state_transitions}
    if trans.get(TEMPLATE_STATE_TRANSITION) != TEMPLATE_STATE_ID:
        raise ExtractionError(
            f"stage 图 transition {TEMPLATE_STATE_TRANSITION!r} 不属于 "
            f"state {TEMPLATE_STATE_ID!r}")
    tensor_ids = {t.stable_id for t in graph.tensors}
    expect_tensor = f"tensor:{TEMPLATE_CACHE_TENSOR_NAME}"
    if expect_tensor not in tensor_ids:
        raise ExtractionError(
            f"stage 图缺模板 cache 张量 {expect_tensor!r}（cache_tensor "
            "字段语义 = 图内张量 name，stable ID = tensor:+name）")
    state_edges = [e for e in graph.edges if e.kind == "state"]
    if not any(e.state_transition_ref == TEMPLATE_STATE_TRANSITION
               for e in state_edges):
        raise ExtractionError(
            f"stage 图无 state 边引用 transition "
            f"{TEMPLATE_STATE_TRANSITION!r}")


def build_dspark_stage_template(
    workspace_root: Path | None = None,
    *,
    values: ModelValues | None = None,
    evidence_base: EvidenceBase | None = None,
) -> DSparkStageBundle:
    """构建公共 stage 语义图（decode 完整链 + prefill passthrough）。

    证据核对 → 图构建 → 通用引用检查 → fold 成员/边界映射核对（成员=
    当前 stage _add_attn_flow 实际返回集合，逐成员与映射端口在当前图
    验证存在，missing 具名拒绝）→ 坐标记录核对。任何失败具名拒绝、
    不给图（EvidenceError / ExtractionError，见模块 docstring）。"""
    root = Path(workspace_root) if workspace_root is not None else find_workspace_root()
    values = values or load_model_values(root)
    evidence_base = evidence_base or load_evidence_base(root)
    registry = EvidenceRegistry(evidence_base, root)
    # 证据同源注册：参考提取（hc/MoE/Block 骨架条目）+ B1 Attention 内部
    # 条目 + 本模块 stage 条目。同 ID 同原文，重复注册即拒绝（防漂移）。
    for ev in _reference_evidence_defs() + _attn_evidence_defs_local() \
            + _stage_evidence_defs():
        registry.register(ev)

    axes = _axes(values)
    b = _GraphBuilder(axes)
    t = _add_tensors(b, axes)
    attn_flow_nodes = _add_flow(b, t)
    _attn_states(b)
    guards = _attn_guards()

    for g in guards:
        for ref in g.evidence_refs:
            if ref not in registry:
                raise ExtractionError(
                    f"guard {g.stable_id} 引用了未注册证据: {ref!r}")
    for ev_id in b.evidence_ids:
        if ev_id not in registry:
            raise ExtractionError(f"图引用了未注册证据: {ev_id!r}")
    for axis in axes.values():
        if (axis.evidence_ref.startswith("ev_")
                and axis.evidence_ref not in registry):
            raise ExtractionError(
                f"轴 {axis.symbol} 引用了未注册证据: {axis.evidence_ref!r}")

    # 边界映射：内部节点/端口必须真实存在于当前 stage 图自身拓扑（fed
    # 源节点带 x 输入端口；B1 图同 ID 节点是图输入源无输入端口——映射
    # 以当前 stage 图为准核对，不从另建 B1 图推断）。
    mappings = _attn_boundary_mappings(t)

    # fold 成员 = 当前 stage 实际 _add_attn_flow 返回的节点集合（本图
    # 真实创建的成员，不是从另建 B1 图拷贝的 ID 清单）；随后逐成员在
    # 当前图验证真实存在，任何 missing 即具名拒绝（不允许 continue
    # 跳过）。
    fold_members = tuple(sorted(set(attn_flow_nodes)))
    b.fold_group("fg_stage_attn", fold_members, mappings)

    graph = b.build(guards, STAGE_TEMPLATE_ID)

    # 构建后核对（当前图）：fold 成员/边界映射逐项验证（missing 具名拒绝）
    _verify_fold_members(graph, fold_members, mappings)
    # 模板 state 常量与真实图核对（state 绑定读回的可信来源）
    _verify_state_facts(graph)

    # 坐标值显式记录（同 B1 语义：坐标值非存储 view；消费端口在本图内可
    # 证伪——cache_update/q_rope 等内部节点真实存在）
    coords = _attn_coords(t, node_map={n.stable_id: n for n in graph.nodes},
                          tensor_ids={x.stable_id for x in graph.tensors})
    cm = build_chain_map(root, values=values)
    # 参数/状态展开绑定表（模板 → 实例；读回一致性由 deserialize_expansion
    # 复检，期望目录来自固定 ModelValues）
    expansion = build_template_expansion(values)
    _verify_expansion(expansion, values)

    return DSparkStageBundle(graph=graph, registry=registry, values=values,
                             coordinate_expressions=coords,
                             boundary_mappings=mappings, chain=cm,
                             expansion=expansion,
                             attention_template_id=ATTENTION_TEMPLATE_ID)


def _attn_evidence_defs_local():
    """B1 Attention 证据条目（同源转发，避免循环依赖的显式接缝）。"""
    from structure_reader.extract.deepseek_v4_dspark import (
        _evidence_defs as _attn_defs,
    )
    return _attn_defs()
