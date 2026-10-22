"""DeepSeek-V4-Flash-0731 DSpark 末出口 head 链（工作包 B · 步骤 B3b 限定恢复02）。

数据来源：models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py
（固定 SHA256 基线，:mod:`structure_reader.extract.evidence` 逐条核对原文）。
静态取证，不导入/不执行模型代码。

本轮范围（限定恢复单 resume_b3b02_head.md——只交付可运行的末出口
head 链；B3 整体仍 in_progress，本模块不是完整 B3/DSpark/V2 交付）：
1. last stage 末出口 head 链（forward_head 860-874 的 hc_head→norm→head
   段）：外部上游 h_out [B,K,N_hc,H]（graph_input 消费 stage2 h_out，
   forward_spec 931-932 逐 stage 串接、935 mtp[-1] 实参）→ last 专有
   hc_head（hc_head_fn/scale/base 参数真实消费边；flatten(2).float() →
   F.linear×rsqrt → sigmoid(mixes*scale+base)+hc_eps 门控 → 加权汇聚
   [B,K,H]——sigmoid 加权 HC 汇聚，非 Sinkhorn）→ pre-norm x [B,K,H]
   （862 行后、863 norm 前的 x）→ last.norm（835，last stage 专有
   RMSNorm）→ 共享 Transformer.head（ParallelHead full_logits=True，
   904 绑定 mtp[-1].head 同一对象非复制）→ fp32 logits [B,K,V]。
2. 角色/相位：实际 3 stage 仅 last（stage2）有出口角色（834 分支），
   复用 B2 derive_stage_bindings 纯角色推导（只读复用，单 stage
   first&last 为合成测试，不伪装修改固定 3 stage 输入）；prefill
   （start_pos==0）在 forward_spec 933-934 stage loop 后 return——
   整条 head 链 prefill 不活动、decode 活动，全部边挂机器可检
   g_decode（start_pos > 0，evaluate_bool 可求值）。
3. 最小可检角色/参数映射（ExitRoleMap canonical JSON）：guard 表达式
   机器求值核对 + 各 stage 角色与 B2 推导逐字段核对 + 出口参数路径 +
   共享 head 身份；读回篡改具名拒绝。
4. 篡改拒绝：对真实序列化图做引用合法的实质负例（删必要 head 边、
   一致重接、共享 head 身份复制、prefill 误启去 guard）——先证
   validate_graph_references 为 0 再断言语义具名拒绝。
   fix01 补修追加三处已复现漏洞：必要节点 op_type/granularity/端口
   闭世界（sigmoid_gate 非 Sinkhorn）、消费轴可信值/表达式核对
   （V=129280 同步改值具名拒绝）、n_stages 类型与 B2 派生长度核对。

不做（本轮明确未实现，后续恢复单续接；不得把本模块当完整 B3b）：
Markov 递推（866-871 markov_head/sample/output_ids 版本链）、
Confidence（807-815/872-873）、forward_head 返回三元组、B2 内部图
pending 关闭、B4 全量集成。图内无 Markov/Confidence 占位节点——
不把未实现链画成已实现。

可调用入口：
- build_dspark_exit_graph() → DSparkExitBundle（末出口 head 链图 +
  证据注册表（B1/B2 同源 + 本包 ev_ex_*）+ ModelValues + B2 角色推导）
- verify_exit_graph(graph, values)（构建期已调用，测试/读回复用）
- build_exit_role_map(values) → 最小可检角色/参数映射
- serialize_exit_role_map / deserialize_exit_role_map（canonical JSON
  往返，读回全量核对 + guard 机器求值核对）

失败异常接口（以实际接口为准）：证据核对失败抛
structure_reader.extract.evidence.EvidenceError；图构造错误、
validate_graph_references 未通过、角色推导不一致、角色映射 JSON 结构
不合法或语义篡改抛
structure_reader.extract.deepseek_v4_reference.ExtractionError
（ValueError 子类）。均具名拒绝、失败不给图/不给映射。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.deepseek_v4_dspark import (
    _axes as _attn_axes,
    _evidence_defs as _attn_evidence_defs,
)
from structure_reader.extract.deepseek_v4_reference import (
    ExtractionError,
    _GraphBuilder,
)
from structure_reader.extract.dspark_stage import (
    _stage_evidence_defs,
    derive_stage_bindings,
)
from structure_reader.extract.evidence import (
    EvidenceBase,
    EvidenceRegistry,
    SourceEvidence,
    load_evidence_base,
)
from structure_reader.graph_document import SemanticGraph
from structure_reader.graph_ir import Guard, NamedAxis
from structure_reader.model_registry import find_workspace_root
from structure_reader.rules.deepseek_v4_flash_0731 import (
    AXIS_DEFS,
    ModelValues,
    load_model_values,
)
from structure_reader.scenario_evaluator import evaluate_bool

__all__ = [
    "EXIT_TEMPLATE_ID",
    "EXIT_SCHEMA",
    "G_DECODE_ID",
    "G_DECODE_EXPR",
    "DSparkExitBundle",
    "verify_exit_graph",
    "build_dspark_exit_graph",
    "build_exit_role_map",
    "serialize_exit_role_map",
    "deserialize_exit_role_map",
]

EXIT_TEMPLATE_ID = "deepseek_v4_flash_0731:dspark_exit"
EXIT_SCHEMA = "dspark-exit-role-map/v1"

# 末出口整条 head 链相位 guard：forward_spec 933-934 prefill
# （start_pos==0）在 stage loop 后 return——head 链只在 decode 活动。
# 表达式与 B1 g_decode 同文（deepseek_v4_dspark.py:769）。
G_DECODE_ID = "g_decode"
G_DECODE_EXPR = "start_pos > 0"


# ---------------------------------------------------------------------------
# 证据目录：本包新增条目（ev_ex_*，仅 head 链所需）。hc/Attention/stage
# 等既有条目复用 B1/B2 同源注册（同 ID 同原文，重复注册即拒绝）。
# ---------------------------------------------------------------------------

_E = SourceEvidence


def _exit_evidence_defs() -> tuple[SourceEvidence, ...]:
    m = "model.py"
    return (
        _E("ev_ex_hc_head", m, "Block.hc_head (last-stage exit)", 709, 716,
           "def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):\n"
           "    shape, dtype = x.size(), x.dtype\n"
           "    x = x.flatten(2).float()\n"
           "    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)\n"
           "    mixes = F.linear(x, hc_fn) * rsqrt\n"
           "    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps\n"
           "    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)\n"
           "    return y.to(dtype)",
           "末出口 hc_head（last 专有调用 model.py:862，参数 mtp.last."
           "hc_head_fn/scale/base）：[B,K,N_hc,H] flatten(2)→float（711）"
           "→ rsqrt 归一化因子（712）→ F.linear(hc_fn)*rsqrt mixes（713）"
           "→ sigmoid(mixes*scale+base)+hc_eps 门控（714）→ "
           "sum(pre.unsqueeze(-1)*x.view(shape),dim=2) 加权汇聚 [B,K,H]"
           "（715）→ y.to(dtype) 回转输入 dtype（716）。sigmoid 门控加权"
           "汇聚，非 Sinkhorn（与 hc_pre 的 sinkhorn_split 机制不同）",
           "source_checked"),
        _E("ev_ex_parallel_head", m, "ParallelHead weight/forward", 719, 740,
           "class ParallelHead(nn.Module):\n"
           "    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):\n"
           "        self.part_vocab_size = (vocab_size // world_size)\n"
           "        # lm_head in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for easier computation of logits later.\n"
           "        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim, dtype=torch.float32))\n"
           "    def forward(self, x: torch.Tensor, full_logits=False):\n"
           "        if not full_logits:\n"
           "            x = x[:, -1]\n"
           "        logits = F.linear(x.float(), self.weight)",
           "共享 head 参数/前向：ParallelHead.weight fp32（729；728 源码"
           "注释原文——checkpoint 存 bf16，与运行参数分开记录）；"
           "part_vocab_size = vocab_size // world_size（727），world_size=1"
           " 分析假设下全宽 [V,H]；forward full_logits=True 不取 x[:,-1]"
           " 全宽输出（731-735；末出口 863 传 True，主干 923 默认 False 取"
           " x[:,-1]）；x.float() fp32 求值，world_size=1 无 all_gather 分支",
           "source_checked"),
        _E("ev_ex_exit_role", m, "last-stage exit branch params", 834, 841,
           "if stage_id == args.n_mtp_layers - 1:\n"
           "    self.norm = RMSNorm(args.dim, args.norm_eps)\n"
           "    self.markov_head = DSparkMarkovHead(args.vocab_size, args.dspark_markov_rank)\n"
           "    self.confidence_head = DSparkConfidenceHead(args.dim + args.dspark_markov_rank)\n"
           "    with set_dtype(torch.float32):\n"
           "        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim))\n"
           "        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult))\n"
           "        self.hc_head_scale = nn.Parameter(torch.empty(1))",
           "出口角色专有性（834 分支）：stage_id == n_mtp_layers-1（本输入"
           " stage2）才实例化 norm（835）与 hc_head_fn/scale/base（838-841，"
           "set_dtype(torch.float32) 块内 → 存储 fp32；scale 是 torch."
           "empty(1) 形状 (1,) 非标量）；markov_head/confidence_head（836/"
           "837）同分支但**本轮未实现**——记录创建事实，图内不画占位节点。"
           "forward_head 只在 mtp[-1] 被调用（forward_spec 935），stage0/1 "
           "无出口角色", "source_checked"),
        _E("ev_ex_shared_head_ref", m, "mtp instances bind shared head", 903, 904,
           "self.mtp[-1].embed = self.embed\n"
           "self.mtp[-1].head = self.head",
           "共享 head 引用：Transformer.head（ParallelHead，897 创建）同一"
           "对象逐实例绑定 mtp[-1].head（903-904 循环体，全部 mtp 实例）——"
           "末出口图单份 head weight 张量 = transformer.head 同一身份，非"
           "复制。主干 923 与 last 863 消费同一 head 对象（不同调用点）",
           "source_checked"),
        _E("ev_ex_spec_return", m, "forward_spec prefill early-return", 929, 936,
           "def forward_spec(self, input_ids: torch.Tensor, main_hidden: torch.Tensor, start_pos: int = 0):\n"
           "    h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)\n"
           "    for layer in self.mtp:\n"
           "        h = layer(h, start_pos, input_ids, main_x)\n"
           "    if start_pos == 0:\n"
           "        return\n"
           "    output_ids, logits, confidence = self.mtp[-1].forward_head(h, input_ids)\n"
           "    return output_ids, logits, confidence",
           "末出口相位证据：forward_spec 933-934 prefill（start_pos==0）在 "
           "stage loop 后 return——935 末出口 forward_head 只在 decode"
           "（start_pos > 0）可达；整条 head 链 decode-only，图内逐边挂 "
           "g_decode（start_pos > 0，同 B1 表达式），不通过删边实现屏蔽",
           "source_checked"),
        _E("ev_ex_exit_premise", m, "exit h_out port premise", None, None,
           None,
           "末出口外部 h 端口来源前提",
           "derived",
           derivation="末出口图输入 h_out = forward_spec 931-932 逐 stage "
                      "串接后的 mtp[-1] h_out [B,K,N_hc,H]（935 实参；K="
                      "N_draft 草稿轴）。不复制外部上游 stage 图，只声明该"
                      "端口来源。933-934 prefill return——本端口消费整链 "
                      "decode-only。forward_head 返回 (output_ids, logits, "
                      "confidence)（874），本轮图只展开 head 链到 logits，"
                      "output_ids/confidence 未实现"),
    )


def _c1_axis() -> NamedAxis:
    """单元宽轴 C1：hc_head_scale = torch.empty(1)（841）形状 (1,)——
    单元宽轴表达，不冒充标量。构图与可信轴核对共用同一定义防漂移。"""
    return NamedAxis(
        "C1", "unit_width",
        "单元宽（shape-(1,) 参数，如 hc_head_scale torch.empty(1)）",
        "推导 model.py:841", "model.py:841", value=1)


def _exit_axes(values: ModelValues) -> dict[str, NamedAxis]:
    """末出口图轴：B1 模块轴同源复制 + 本包两条（V rules 同源、C1 单元
    宽派生）。N_hcH 已在 B1 轴集（rules 同源 expression="N_hc * H"）。"""
    axes = _attn_axes(values)
    by_symbol = {a.symbol: a for a in AXIS_DEFS}
    a = by_symbol["V"]
    axes["V"] = NamedAxis(
        symbol=a.symbol, name=a.name, meaning=a.meaning,
        origin=a.origin, evidence_ref=a.evidence_ref,
        value=a.value, expression=a.expression,
        unknown_reason=a.unknown_reason,
    )
    axes["C1"] = _c1_axis()
    return axes


def _trusted_exit_axes(values: ModelValues) -> dict[str, NamedAxis]:
    """本图消费轴的可信定义（fix01 ②核对基准，不从被检图生成期望）。

    覆盖 _EXPECTED_TENSOR_SHAPES 出现的全部 symbol：V/H/N_hc/N_draft
    取 rules AXIS_DEFS 同源定义并与 ModelValues 对应字段交叉核对（防
    rules/values 漂移）；N_hcH/B 取 rules 表达式轴同源（B 保持场景动态
    表达式语义，value=None，不伪造数值）；C1 本包单元宽轴（_exit_axes
    同一构造）。"""
    by_symbol = {a.symbol: a for a in AXIS_DEFS}

    def copy_def(a: object) -> NamedAxis:
        return NamedAxis(
            symbol=a.symbol, name=a.name, meaning=a.meaning,
            origin=a.origin, evidence_ref=a.evidence_ref,
            value=a.value, expression=a.expression,
            unknown_reason=a.unknown_reason,
        )

    trusted: dict[str, NamedAxis] = {}
    for sym, field in (("V", "vocab_size"), ("H", "dim"),
                       ("N_hc", "hc_mult"), ("N_draft", "dspark_block_size")):
        a = by_symbol[sym]
        v = getattr(values, field)
        if a.value != v:
            raise ExtractionError(
                f"可信轴来源漂移：rules 轴 {sym} value={a.value!r} 与 "
                f"ModelValues.{field}={v!r} 不一致——校验基准不可信，"
                "拒绝构建")
        trusted[sym] = copy_def(a)
    trusted["N_hcH"] = copy_def(by_symbol["N_hcH"])
    trusted["B"] = copy_def(by_symbol["B"])
    trusted["C1"] = _c1_axis()
    return trusted


# ---------------------------------------------------------------------------
# 末出口 head 链图
# ---------------------------------------------------------------------------

# 必要节点闭世界（hc_head 三参数 + flatten/factor/mix/gate/reduce +
# norm + 共享 head + 唯一输出）。
_EXPECTED_NODES: frozenset[str] = frozenset({
    "exit.h_out_src",
    "exit.hc_head_fn_w", "exit.hc_head_scale_w", "exit.hc_head_base_w",
    "exit.hc_head.flatten", "exit.hc_head.factor", "exit.hc_head.mix",
    "exit.hc_head.gate", "exit.hc_head.reduce",
    "exit.norm_w", "exit.norm",
    "exit.head_w", "exit.head",
    "exit.logits_out",
})

# 必要张量闭世界（多余张量 = 独立身份/复制）。
_EXPECTED_TENSORS: frozenset[str] = frozenset({
    "exit:h_out", "exit:hc_flat", "exit:hc_factor", "exit:hc_mixes",
    "exit:hc_gate", "exit:prenorm_x",
    "exit:hc_head_fn_w", "exit:hc_head_scale_w", "exit:hc_head_base_w",
    "exit:norm_w", "exit:norm_x", "exit:head_w", "exit:logits",
})

# 必要生产/消费链闭世界（边 ID → 源节点.端口 → 目的节点.端口 + 张量）。
# 通用引用检查只做三方一致，不查必要链——缺边、保 ID 一致重接、共享
# 参数消费边丢失由此具名拒绝。
_EXPECTED_CHAIN: tuple[tuple[str, str, str, str, str, str], ...] = (
    # 外部上游端口 → hc_head flatten（710-711）
    ("e_ex_hin_flat", "exit.h_out_src", "y", "exit.hc_head.flatten",
     "x", "exit:h_out"),
    # hc_head 内部（711-715）
    ("e_ex_flat_factor", "exit.hc_head.flatten", "x_flat",
     "exit.hc_head.factor", "x", "exit:hc_flat"),
    ("e_ex_flat_mix", "exit.hc_head.flatten", "x_flat",
     "exit.hc_head.mix", "x", "exit:hc_flat"),
    ("e_ex_fnw_mix", "exit.hc_head_fn_w", "y", "exit.hc_head.mix",
     "w", "exit:hc_head_fn_w"),
    ("e_ex_factor_mix", "exit.hc_head.factor", "y", "exit.hc_head.mix",
     "factor", "exit:hc_factor"),
    ("e_ex_mix_gate", "exit.hc_head.mix", "y", "exit.hc_head.gate",
     "x", "exit:hc_mixes"),
    ("e_ex_scale_gate", "exit.hc_head_scale_w", "y", "exit.hc_head.gate",
     "scale", "exit:hc_head_scale_w"),
    ("e_ex_base_gate", "exit.hc_head_base_w", "y", "exit.hc_head.gate",
     "base", "exit:hc_head_base_w"),
    ("e_ex_gate_reduce", "exit.hc_head.gate", "y", "exit.hc_head.reduce",
     "pre", "exit:hc_gate"),
    ("e_ex_flat_reduce", "exit.hc_head.flatten", "x_flat",
     "exit.hc_head.reduce", "x", "exit:hc_flat"),
    # 汇聚输出 = pre-norm x → last.norm（862 行后、863 norm 前）
    ("e_ex_prenorm_norm", "exit.hc_head.reduce", "y", "exit.norm",
     "x", "exit:prenorm_x"),
    ("e_ex_normw_norm", "exit.norm_w", "y", "exit.norm",
     "w", "exit:norm_w"),
    # last.norm → 共享 head（863 full_logits=True）
    ("e_ex_norm_head", "exit.norm", "y", "exit.head",
     "x", "exit:norm_x"),
    ("e_ex_headw_head", "exit.head_w", "y", "exit.head",
     "w", "exit:head_w"),
    # 唯一输出（936 logits 返回；output_ids/confidence 未实现不画）
    ("e_ex_logits_out", "exit.head", "y", "exit.logits_out",
     "x", "exit:logits"),
)

# 张量 dtype/shape 闭世界核对表（name → dtype + 有序轴）。
_EXPECTED_TENSOR_SHAPES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("exit:h_out", "bf16", ("B", "N_draft", "N_hc", "H")),
    ("exit:hc_flat", "fp32", ("B", "N_draft", "N_hcH")),
    ("exit:hc_factor", "fp32", ("B", "N_draft", "C1")),
    ("exit:hc_mixes", "fp32", ("B", "N_draft", "N_hc")),
    ("exit:hc_gate", "fp32", ("B", "N_draft", "N_hc")),
    ("exit:prenorm_x", "bf16", ("B", "N_draft", "H")),
    ("exit:hc_head_fn_w", "fp32", ("N_hc", "N_hcH")),
    ("exit:hc_head_scale_w", "fp32", ("C1",)),
    ("exit:hc_head_base_w", "fp32", ("N_hc",)),
    ("exit:norm_w", "fp32", ("H",)),
    ("exit:norm_x", "bf16", ("B", "N_draft", "H")),
    ("exit:head_w", "fp32", ("V", "H")),
    ("exit:logits", "fp32", ("B", "N_draft", "V")),
)


# 必要节点算子/粒度/端口闭世界（可信常量，与 _add_exit_flow 构建同文；
# 不从被检图生成期望）。端口 (名称, 张量名) 有序元组，张量名经
# tensors[name].stable_id 解析。引用检查只查节点 ID/连线，同 ID 下换
# 关键算子（hc_head 门控 sigmoid_gate→Sinkhorn）/换粒度/动端口面均
# 引用合法——由此逐节点固定核对，具名拒绝。
_EXPECTED_NODE_OPS: dict[
    str,
    tuple[str, str,
          tuple[tuple[str, str], ...],
          tuple[tuple[str, str], ...]],
] = {
    "exit.h_out_src": ("graph_input", "tensor_op", (),
                       (("y", "exit:h_out"),)),
    "exit.hc_head_fn_w": ("hc_fn_param", "parameter_op", (),
                          (("y", "exit:hc_head_fn_w"),)),
    "exit.hc_head_scale_w": ("hc_scale_param", "parameter_op", (),
                             (("y", "exit:hc_head_scale_w"),)),
    "exit.hc_head_base_w": ("hc_base_param", "parameter_op", (),
                            (("y", "exit:hc_head_base_w"),)),
    "exit.hc_head.flatten": ("flatten", "tensor_op",
                             (("x", "exit:h_out"),),
                             (("x_flat", "exit:hc_flat"),)),
    "exit.hc_head.factor": ("rms_factor", "tensor_op",
                            (("x", "exit:hc_flat"),),
                            (("y", "exit:hc_factor"),)),
    "exit.hc_head.mix": ("linear", "parameter_op",
                         (("x", "exit:hc_flat"), ("w", "exit:hc_head_fn_w"),
                          ("factor", "exit:hc_factor")),
                         (("y", "exit:hc_mixes"),)),
    "exit.hc_head.gate": ("sigmoid_gate", "tensor_op",
                          (("x", "exit:hc_mixes"),
                           ("scale", "exit:hc_head_scale_w"),
                           ("base", "exit:hc_head_base_w")),
                          (("y", "exit:hc_gate"),)),
    "exit.hc_head.reduce": ("weighted_sum", "tensor_op",
                            (("pre", "exit:hc_gate"), ("x", "exit:hc_flat")),
                            (("y", "exit:prenorm_x"),)),
    "exit.norm_w": ("rms_norm_weight", "parameter_op", (),
                    (("y", "exit:norm_w"),)),
    "exit.norm": ("rms_norm", "parameter_op",
                  (("x", "exit:prenorm_x"), ("w", "exit:norm_w")),
                  (("y", "exit:norm_x"),)),
    "exit.head_w": ("linear_weight", "parameter_op", (),
                    (("y", "exit:head_w"),)),
    "exit.head": ("linear", "parameter_op",
                  (("x", "exit:norm_x"), ("w", "exit:head_w")),
                  (("y", "exit:logits"),)),
    "exit.logits_out": ("graph_output", "boundary_op",
                        (("x", "exit:logits"),), (("y", "exit:logits"),)),
}


def _add_exit_tensors(b: _GraphBuilder, axes: dict[str, NamedAxis]
                      ) -> dict[str, str]:
    """末出口 head 链张量角色 → stable_id（前缀 exit:）。

    dtype 依据：last h_out bf16（stage 链 bf16）；hc_head 内部 flatten/
    float 后 fp32 求值（711-715），汇聚输出 y.to(dtype) 回 bf16（716）；
    pre-norm x / norm(x) bf16（RMSNorm forward 回转 x.dtype）；hc_head
    参数 fp32（838-841 set_dtype(torch.float32) 块，scale 形状 (1,) 用
    C1 单元宽轴）；last.norm weight fp32（RMSNorm 194-195，checkpoint
    存 bf16 分开记录）；head weight fp32（ParallelHead 729，checkpoint
    bf16 注释分开记录）；logits fp32（F.linear(x.float(), fp32 weight)）。"""
    t: dict[str, str] = {}

    def add(role: str, dtype: str, syms: tuple[str, ...],
            evidence: tuple[str, ...] = ()) -> str:
        tid = b.tensor(f"exit:{role}", dtype, syms, evidence)
        t[role] = tid
        return tid

    # 外部上游端口：last stage h_out（forward_spec 931-932 串接 → 935）
    add("h_out", "bf16", ("B", "N_draft", "N_hc", "H"),
        ("ev_ex_exit_premise",))
    # hc_head 内部（711-715）
    add("hc_flat", "fp32", ("B", "N_draft", "N_hcH"), ("ev_ex_hc_head",))
    add("hc_factor", "fp32", ("B", "N_draft", "C1"), ("ev_ex_hc_head",))
    add("hc_mixes", "fp32", ("B", "N_draft", "N_hc"), ("ev_ex_hc_head",))
    add("hc_gate", "fp32", ("B", "N_draft", "N_hc"), ("ev_ex_hc_head",))
    # 汇聚输出 = pre-norm x（862 行后、863 norm 前的 x）
    add("prenorm_x", "bf16", ("B", "N_draft", "H"), ("ev_ex_hc_head",))
    # hc_head 参数（838-841，last 专有 fp32）
    add("hc_head_fn_w", "fp32", ("N_hc", "N_hcH"), ("ev_ex_exit_role",))
    add("hc_head_scale_w", "fp32", ("C1",), ("ev_ex_exit_role",))
    add("hc_head_base_w", "fp32", ("N_hc",), ("ev_ex_exit_role",))
    # last.norm（835；RMSNorm weight fp32）
    add("norm_w", "fp32", ("H",), ("ev_ex_exit_role",))
    add("norm_x", "bf16", ("B", "N_draft", "H"), ())
    # 共享 head（ParallelHead fp32 weight；checkpoint bf16 分开记录）
    add("head_w", "fp32", ("V", "H"),
        ("ev_ex_parallel_head", "ev_ex_shared_head_ref"))
    add("logits", "fp32", ("B", "N_draft", "V"), ("ev_ex_parallel_head",))
    return t


def _add_exit_flow(b: _GraphBuilder, t: dict[str, str]) -> None:
    """末出口 head 链流（hc_head 709-716 + forward_head 860-863 的
    hc_head→norm→head 段 + forward_spec 929-936 出口部分）。

    相位规则：整条 head 链全部边挂可求值 g_decode（start_pos > 0）——
    forward_spec 933-934 prefill 在 stage loop 后 return，935 末出口
    prefill 不可达。参数身份节点静态存在，其到 decode-only 算子的消费
    边同样挂 g_decode（B1 惯例）。不通过删边实现屏蔽。"""
    guard = G_DECODE_ID

    def node(sid, op, gr, inputs=(), outputs=(), ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.node(sid, op, gr, inputs, outputs, ev)

    def edge(sid, src, dst, role, ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.edge(sid, src, dst, t[role], "data", guard_ref=guard,
                      evidence=ev)

    # ===== 外部上游端口（forward_spec 935 h 实参；不复制上游 stage 图）=====
    node("exit.h_out_src", "graph_input", "tensor_op",
         (), (("y", t["h_out"]),), ("ev_ex_exit_premise",))

    # ===== hc_head（862；sigmoid 门控加权汇聚，非 Sinkhorn）=====
    node("exit.hc_head_fn_w", "hc_fn_param", "parameter_op",
         (), (("y", t["hc_head_fn_w"]),), ("ev_ex_exit_role",))
    node("exit.hc_head_scale_w", "hc_scale_param", "parameter_op",
         (), (("y", t["hc_head_scale_w"]),), ("ev_ex_exit_role",))
    node("exit.hc_head_base_w", "hc_base_param", "parameter_op",
         (), (("y", t["hc_head_base_w"]),), ("ev_ex_exit_role",))
    node("exit.hc_head.flatten", "flatten", "tensor_op",
         (("x", t["h_out"]),), (("x_flat", t["hc_flat"]),),
         ("ev_ex_hc_head",))
    edge("e_ex_hin_flat", ("exit.h_out_src", "y"),
         ("exit.hc_head.flatten", "x"), "h_out", ("ev_ex_exit_premise",))
    # rsqrt 归一化因子（712）
    node("exit.hc_head.factor", "rms_factor", "tensor_op",
         (("x", t["hc_flat"]),), (("y", t["hc_factor"]),),
         ("ev_ex_hc_head",))
    edge("e_ex_flat_factor", ("exit.hc_head.flatten", "x_flat"),
         ("exit.hc_head.factor", "x"), "hc_flat", ("ev_ex_hc_head",))
    # mixes = F.linear(flat, hc_fn) * rsqrt（713）
    node("exit.hc_head.mix", "linear", "parameter_op",
         (("x", t["hc_flat"]), ("w", t["hc_head_fn_w"]),
          ("factor", t["hc_factor"])),
         (("y", t["hc_mixes"]),), ("ev_ex_hc_head",))
    edge("e_ex_flat_mix", ("exit.hc_head.flatten", "x_flat"),
         ("exit.hc_head.mix", "x"), "hc_flat", ("ev_ex_hc_head",))
    edge("e_ex_fnw_mix", ("exit.hc_head_fn_w", "y"),
         ("exit.hc_head.mix", "w"), "hc_head_fn_w", ("ev_ex_hc_head",))
    edge("e_ex_factor_mix", ("exit.hc_head.factor", "y"),
         ("exit.hc_head.mix", "factor"), "hc_factor", ("ev_ex_hc_head",))
    # 门控（714）：sigmoid(mixes*scale + base) + hc_eps
    node("exit.hc_head.gate", "sigmoid_gate", "tensor_op",
         (("x", t["hc_mixes"]), ("scale", t["hc_head_scale_w"]),
          ("base", t["hc_head_base_w"])),
         (("y", t["hc_gate"]),), ("ev_ex_hc_head",))
    edge("e_ex_mix_gate", ("exit.hc_head.mix", "y"),
         ("exit.hc_head.gate", "x"), "hc_mixes", ("ev_ex_hc_head",))
    edge("e_ex_scale_gate", ("exit.hc_head_scale_w", "y"),
         ("exit.hc_head.gate", "scale"), "hc_head_scale_w",
         ("ev_ex_hc_head",))
    edge("e_ex_base_gate", ("exit.hc_head_base_w", "y"),
         ("exit.hc_head.gate", "base"), "hc_head_base_w",
         ("ev_ex_hc_head",))
    # 加权汇聚（715-716）：sum(gate.unsqueeze(-1)*x.view(shape),dim=2)
    # → pre-norm x [B,K,H]（y.to(dtype) 回 bf16）
    node("exit.hc_head.reduce", "weighted_sum", "tensor_op",
         (("pre", t["hc_gate"]), ("x", t["hc_flat"])),
         (("y", t["prenorm_x"]),), ("ev_ex_hc_head",))
    edge("e_ex_gate_reduce", ("exit.hc_head.gate", "y"),
         ("exit.hc_head.reduce", "pre"), "hc_gate", ("ev_ex_hc_head",))
    edge("e_ex_flat_reduce", ("exit.hc_head.flatten", "x_flat"),
         ("exit.hc_head.reduce", "x"), "hc_flat", ("ev_ex_hc_head",))

    # ===== last.norm → 共享 head（863；full_logits=True）=====
    node("exit.norm_w", "rms_norm_weight", "parameter_op",
         (), (("y", t["norm_w"]),), ("ev_ex_exit_role",))
    node("exit.norm", "rms_norm", "parameter_op",
         (("x", t["prenorm_x"]), ("w", t["norm_w"])),
         (("y", t["norm_x"]),), ("ev_ex_exit_role",))
    edge("e_ex_prenorm_norm", ("exit.hc_head.reduce", "y"),
         ("exit.norm", "x"), "prenorm_x", ("ev_ex_exit_role",))
    edge("e_ex_normw_norm", ("exit.norm_w", "y"),
         ("exit.norm", "w"), "norm_w", ("ev_ex_exit_role",))
    node("exit.head_w", "linear_weight", "parameter_op",
         (), (("y", t["head_w"]),),
         ("ev_ex_parallel_head", "ev_ex_shared_head_ref"))
    node("exit.head", "linear", "parameter_op",
         (("x", t["norm_x"]), ("w", t["head_w"])),
         (("y", t["logits"]),), ("ev_ex_parallel_head",))
    edge("e_ex_norm_head", ("exit.norm", "y"),
         ("exit.head", "x"), "norm_x", ("ev_ex_parallel_head",))
    edge("e_ex_headw_head", ("exit.head_w", "y"),
         ("exit.head", "w"), "head_w",
         ("ev_ex_parallel_head", "ev_ex_shared_head_ref"))

    # ===== 唯一输出（936 logits；output_ids/confidence 未实现不画）=====
    node("exit.logits_out", "graph_output", "boundary_op",
         (("x", t["logits"]),), (("y", t["logits"]),),
         ("ev_ex_parallel_head",))
    edge("e_ex_logits_out", ("exit.head", "y"),
         ("exit.logits_out", "x"), "logits", ("ev_ex_parallel_head",))


# ---------------------------------------------------------------------------
# 角色推导（B2 只读复用）与最小可检角色/参数映射
# ---------------------------------------------------------------------------

# last stage 出口模块（model.py 834-841 分支 + forward_head；markov_head/
# confidence_head 同分支创建但本轮未实现，不入 exit_modules）。
_EXIT_MODULES = ("norm", "hc_head_fn", "hc_head_base", "hc_head_scale",
                 "forward_head")

_SCOPE_IMPLEMENTED = (
    "hc_head（sigmoid 门控加权汇聚，非 Sinkhorn）",
    "last.norm",
    "shared head（full_logits=True）",
)
_SCOPE_PENDING = ("markov_head", "confidence_head", "sample", "B4 集成")

_PHASE_PREFILL_REASON = (
    "forward_spec 933-934 prefill（start_pos==0）在 stage loop 后 return，"
    "935 末出口 prefill 不可达")
_PHASE_DECODE_REASON = "start_pos > 0 才到达 forward_spec 935 末出口 forward_head"


def _derive_and_check_roles(values: ModelValues):
    """B2 derive_stage_bindings 只读复用 + 出口角色一致性核对。

    实际固定输入 3 stage 仅 last 有出口角色（model.py 834 分支）；
    n_mtp_layers==1 时纯逻辑分支单 stage 同时 first&last（B2 已覆盖，
    合成测试使用，不修改固定输入）。"""
    roles = derive_stage_bindings(values.n_mtp_layers, values.n_layers)
    last = roles[-1]
    if not last.is_last:
        raise ExtractionError(
            f"B2 角色推导末 stage（{last.param_path}）is_last 不成立——"
            "出口角色推导与 model.py 834 分支不一致")
    for module in _EXIT_MODULES:
        if module not in last.role_modules:
            raise ExtractionError(
                f"B2 角色推导末 stage 缺出口模块 {module!r}（role_modules="
                f"{last.role_modules}）——与 model.py 834-841/862-863 不一致")
    for r in roles[:-1]:
        if "forward_head" in r.role_modules:
            raise ExtractionError(
                f"非末 stage {r.param_path} 出现出口角色 forward_head——"
                "834 分支只在 stage_id == n_mtp_layers-1 实例化出口模块")
    return roles


def build_exit_role_map(values: ModelValues) -> dict:
    """最小可检角色/参数映射（canonical JSON 往返，deserialize 全量核对）。

    只覆盖本轮 head 链：guard 机器求值、各 stage 角色与 B2 推导一致、
    出口参数路径、共享 head 身份、scope 边界（pending 显式列出未实现
    的 Markov/Confidence——把 pending 删掉即映射不合法，防 overclaim）。"""
    roles = _derive_and_check_roles(values)
    last = roles[-1]
    sid = last.stage_id
    return {
        "schema": EXIT_SCHEMA,
        "template_id": EXIT_TEMPLATE_ID,
        "guard": {
            "id": G_DECODE_ID,
            "expr": G_DECODE_EXPR,
            "prefill_active": False,
            "decode_active": True,
        },
        "n_stages": len(roles),
        "stages": [
            {
                "stage_id": r.stage_id,
                "actual_layer_id": r.actual_layer_id,
                "param_path": r.param_path,
                "is_first": r.is_first,
                "is_last": r.is_last,
                "exit_role": r.is_last,
                "role_modules": list(r.role_modules),
            }
            for r in roles
        ],
        "exit_stage": {
            "stage_id": sid,
            "param_path": last.param_path,
            "exit_modules": list(_EXIT_MODULES),
            "params": {
                "hc_head_fn": f"mtp.{sid}.hc_head_fn",
                "hc_head_scale": f"mtp.{sid}.hc_head_scale",
                "hc_head_base": f"mtp.{sid}.hc_head_base",
                "norm": f"mtp.{sid}.norm",
                "head": f"shared:Transformer.head=mtp.{sid}.head"
                        "（904 同一对象非复制）",
            },
        },
        "phases": {
            "prefill": {"head_chain_active": False,
                        "reason": _PHASE_PREFILL_REASON},
            "decode": {"head_chain_active": True,
                       "reason": _PHASE_DECODE_REASON},
        },
        "scope": {
            "implemented": list(_SCOPE_IMPLEMENTED),
            "pending": list(_SCOPE_PENDING),
        },
    }


def serialize_exit_role_map(role_map: dict) -> str:
    """canonical JSON（sort_keys，紧凑分隔符，ensure_ascii=False）。"""
    return json.dumps(role_map, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _reject(what: str, got, want) -> ExtractionError:
    return ExtractionError(
        f"末出口角色映射 {what} 不符：得到 {got!r}，期望 {want!r}"
        "——语义篡改具名拒绝")


def deserialize_exit_role_map(text: str, values: ModelValues) -> dict:
    """读回全量核对：结构闭世界 + n_stages 类型/B2 派生长度核对（fix01 ③）
    + B2 角色逐字段重推导核对 + guard 表达式机器求值核对 + scope 边界
    （pending 缺失即拒绝）。任何不符具名拒绝。"""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"末出口角色映射 JSON 不合法: {exc}") from exc
    if not isinstance(doc, dict):
        raise ExtractionError("末出口角色映射必须是 JSON 对象")
    expected_top = {"schema", "template_id", "guard", "n_stages", "stages",
                    "exit_stage", "phases", "scope"}
    got_top = set(doc)
    if got_top != expected_top:
        raise ExtractionError(
            f"末出口角色映射顶层键闭世界不符：多 {sorted(got_top - expected_top)}"
            f" 缺 {sorted(expected_top - got_top)}")
    if doc["schema"] != EXIT_SCHEMA:
        raise _reject("schema", doc["schema"], EXIT_SCHEMA)
    if doc["template_id"] != EXIT_TEMPLATE_ID:
        raise _reject("template_id", doc["template_id"], EXIT_TEMPLATE_ID)

    # guard：字段闭世界 + 表达式一致 + 机器求值活动性一致
    guard = doc["guard"]
    if not isinstance(guard, dict) or set(guard) != {
            "id", "expr", "prefill_active", "decode_active"}:
        raise ExtractionError(
            f"末出口角色映射 guard 字段闭世界不符: {guard!r}")
    if guard["id"] != G_DECODE_ID:
        raise _reject("guard.id", guard["id"], G_DECODE_ID)
    if guard["expr"] != G_DECODE_EXPR:
        raise _reject("guard.expr", guard["expr"], G_DECODE_EXPR)
    if guard["prefill_active"] is not False or guard["decode_active"] is not True:
        raise ExtractionError(
            f"末出口角色映射 guard 活动性被改: {guard!r}——prefill"
            "（start_pos==0）在 forward_spec 933-934 return，head 链"
            " prefill 不活动")
    if evaluate_bool(guard["expr"], {"start_pos": 0}) is not False:
        raise ExtractionError(
            f"guard 表达式 {guard['expr']!r} 在 start_pos=0 机器求值非 False"
            "——与 prefill_active=false 不一致")
    if evaluate_bool(guard["expr"], {"start_pos": 1}) is not True:
        raise ExtractionError(
            f"guard 表达式 {guard['expr']!r} 在 start_pos=1 机器求值非 True"
            "——与 decode_active=true 不一致")

    # stages：与 B2 derive_stage_bindings 重推导逐字段核对
    roles = _derive_and_check_roles(values)

    # n_stages（fix01 ③）：类型必须是 int（bool/float/str 拒绝）且等于
    # B2 派生 stage 数——顶层 n_stages 改 99 而 stages/exit_stage 原值时
    # 此前无核对放行。错误具名拒绝。
    n_stages = doc["n_stages"]
    if isinstance(n_stages, bool) or not isinstance(n_stages, int):
        raise ExtractionError(
            f"末出口角色映射 n_stages 类型不符: {n_stages!r}"
            f"（{type(n_stages).__name__}）应为 int——类型篡改具名拒绝")
    if n_stages != len(roles):
        raise _reject("n_stages", n_stages, len(roles))

    stages = doc["stages"]
    if not isinstance(stages, list) or len(stages) != len(roles):
        raise ExtractionError(
            f"末出口角色映射 stages 数量 {len(stages) if isinstance(stages, list) else stages!r}"
            f" 应为 {len(roles)}（n_mtp_layers={values.n_mtp_layers}）")
    stage_keys = {"stage_id", "actual_layer_id", "param_path", "is_first",
                  "is_last", "exit_role", "role_modules"}
    for r, s in zip(roles, stages):
        if not isinstance(s, dict) or set(s) != stage_keys:
            raise ExtractionError(
                f"末出口角色映射 stage 记录字段闭世界不符: {s!r}")
        want = {
            "stage_id": r.stage_id,
            "actual_layer_id": r.actual_layer_id,
            "param_path": r.param_path,
            "is_first": r.is_first,
            "is_last": r.is_last,
            "exit_role": r.is_last,
            "role_modules": list(r.role_modules),
        }
        for key, want_v in want.items():
            if s[key] != want_v:
                raise _reject(f"stages[{r.stage_id}].{key}", s[key], want_v)

    # exit_stage：只允许 last
    exit_stage = doc["exit_stage"]
    last = roles[-1]
    sid = last.stage_id
    want_exit = {
        "stage_id": sid,
        "param_path": last.param_path,
        "exit_modules": list(_EXIT_MODULES),
        "params": {
            "hc_head_fn": f"mtp.{sid}.hc_head_fn",
            "hc_head_scale": f"mtp.{sid}.hc_head_scale",
            "hc_head_base": f"mtp.{sid}.hc_head_base",
            "norm": f"mtp.{sid}.norm",
            "head": f"shared:Transformer.head=mtp.{sid}.head"
                    "（904 同一对象非复制）",
        },
    }
    if not isinstance(exit_stage, dict) or set(exit_stage) != set(want_exit):
        raise ExtractionError(
            f"末出口角色映射 exit_stage 字段闭世界不符: {exit_stage!r}")
    for key, want_v in want_exit.items():
        if exit_stage[key] != want_v:
            raise _reject(f"exit_stage.{key}", exit_stage[key], want_v)

    # phases：与 guard 表达式机器求值一致
    phases = doc["phases"]
    want_phases = {
        "prefill": {"head_chain_active": False,
                    "reason": _PHASE_PREFILL_REASON},
        "decode": {"head_chain_active": True,
                   "reason": _PHASE_DECODE_REASON},
    }
    if not isinstance(phases, dict) or set(phases) != {"prefill", "decode"}:
        raise ExtractionError(
            f"末出口角色映射 phases 字段闭世界不符: {phases!r}")
    for phase, want_v in want_phases.items():
        got = phases[phase]
        if not isinstance(got, dict) or set(got) != set(want_v):
            raise ExtractionError(
                f"末出口角色映射 phases.{phase} 字段闭世界不符: {got!r}")
        for key, want_kv in want_v.items():
            if got[key] is not want_kv and got[key] != want_kv:
                raise _reject(f"phases.{phase}.{key}", got[key], want_kv)

    # scope：边界闭世界——pending 缺失（如删掉 markov_head）即拒绝，
    # 防把未实现链声称为已实现/范围外。
    scope = doc["scope"]
    want_scope = {"implemented": list(_SCOPE_IMPLEMENTED),
                  "pending": list(_SCOPE_PENDING)}
    if not isinstance(scope, dict) or set(scope) != {"implemented", "pending"}:
        raise ExtractionError(
            f"末出口角色映射 scope 字段闭世界不符: {scope!r}")
    for key, want_v in want_scope.items():
        if scope[key] != want_v:
            raise _reject(f"scope.{key}", scope[key], want_v)
    return doc


# ---------------------------------------------------------------------------
# 构建入口与图级语义核对
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DSparkExitBundle:
    """B3b 限定恢复02 交付包：末出口 head 链图 + 证据注册表 + 模型数值
    + B2 角色推导。"""
    graph: SemanticGraph
    registry: EvidenceRegistry
    values: ModelValues
    roles: tuple


def build_dspark_exit_graph(
    workspace_root: Path | None = None,
    *,
    values: ModelValues | None = None,
    evidence_base: EvidenceBase | None = None,
) -> DSparkExitBundle:
    """构建末出口 head 链语义图与角色映射（静态取证，不导入/不执行模型
    代码）。

    证据核对 → 同源注册（B1 Attention + B2 stage + 本包 ev_ex_*；B3a
    entry 证据不被本图引用，不注册）→ 图构建 → 通用引用检查 →
    verify_exit_graph 语义核对 → B2 角色推导一致性核对。任何失败具名
    拒绝、不给图（EvidenceError / ExtractionError）。"""
    root = Path(workspace_root) if workspace_root is not None \
        else find_workspace_root()
    values = values or load_model_values(root)
    evidence_base = evidence_base or load_evidence_base(root)
    registry = EvidenceRegistry(evidence_base, root)
    for ev in _attn_evidence_defs() + _stage_evidence_defs() \
            + _exit_evidence_defs():
        registry.register(ev)

    axes = _exit_axes(values)
    for axis in axes.values():
        if axis.evidence_ref.startswith("ev_") \
                and axis.evidence_ref not in registry:
            raise ExtractionError(
                f"轴 {axis.symbol} 引用了未注册证据: {axis.evidence_ref!r}")

    b = _GraphBuilder(axes)
    t = _add_exit_tensors(b, axes)
    _add_exit_flow(b, t)

    for ev_id in b.evidence_ids:
        if ev_id not in registry:
            raise ExtractionError(f"图引用了未注册证据: {ev_id!r}")

    guards = (Guard(G_DECODE_ID, G_DECODE_EXPR, ("ev_ex_spec_return",)),)
    graph = b.build(guards, EXIT_TEMPLATE_ID)
    verify_exit_graph(graph, values)

    roles = _derive_and_check_roles(values)
    return DSparkExitBundle(graph=graph, registry=registry, values=values,
                            roles=roles)


def verify_exit_graph(graph: SemanticGraph, values: ModelValues) -> None:
    """末出口 head 链图语义核对（构建期调用；测试/读回复用）。

    结构/身份/必要链/相位逐项核对，语义篡改（引用合法但语义错）具名
    拒绝；篡改负例应先证 validate_graph_references 为 0 再断言本函数
    拒绝（引用合法前提下的语义拒绝，非悬空引用兜底）：
    - 相位：唯一 guard g_decode（表达式原文一致、机器求值 start_pos=0
      为 False / 1 为 True）；任何边未挂 g_decode 即 prefill 误启具名
      拒绝（forward_spec 933-934 prefill 在 stage loop 后 return）；
    - 必要节点/张量/生产消费链闭世界：head 链必要边缺失、保 ID 一致
      重接（源/目的端口/张量任一不符）、必要链之外多余节点/张量/边
      （共享 head 复制身份、第二 head 通路）——全部具名拒绝；
    - 必要节点算子/粒度/端口面闭世界（fix01 ①）：op_type/granularity/
      输入输出端口对 _EXPECTED_NODE_OPS 可信常量逐节点核对——同 ID 换
      关键算子（hc_head 门控 sigmoid_gate→Sinkhorn）具名拒绝；
    - dtype/shape 逐张量核对（h_out bf16 [B,K,N_hc,H]、head_w fp32
      [V,H]、logits fp32 [B,K,V]、hc 参数 fp32、scale (C1,) 非标量）；
    - 消费轴可信值/表达式核对（fix01 ②）：named_axes 中本图消费轴
      （V/H/N_hc/N_draft/N_hcH/B/C1）对照 rules AXIS_DEFS + ModelValues
      交叉派生的可信定义逐字段核对，named_axes 与张量 axes 同步改值
      （V 129280→129281）具名拒绝；B 保持场景动态表达式（不伪造数值）；
    - 共享 head 单一身份：head_w 恰一、消费边仅 exit.head.w、head 节点
      与 head_w 节点携带 ParallelHead/共享引用证据；
    - 唯一输出：恰一 graph_output 节点消费 exit:logits（来自
      exit.head.y）。"""
    if graph.template_id != EXIT_TEMPLATE_ID:
        raise ExtractionError(
            f"末出口图 template_id 不符: {graph.template_id!r}")

    # 相位：唯一 g_decode + 表达式一致 + 机器求值
    guards = {g.stable_id: g for g in graph.guards}
    if set(guards) != {G_DECODE_ID}:
        raise ExtractionError(
            f"末出口图 guards 应恰为 [{G_DECODE_ID}]，得到 {sorted(guards)}"
            "——head 链 decode-only 单相位条件")
    g = guards[G_DECODE_ID]
    if g.predicate_expr != G_DECODE_EXPR:
        raise ExtractionError(
            f"g_decode 表达式被改: {g.predicate_expr!r} 应为 "
            f"{G_DECODE_EXPR!r}（forward_spec 933-934 prefill return）")
    if evaluate_bool(G_DECODE_EXPR, {"start_pos": 0}) is not False \
            or evaluate_bool(G_DECODE_EXPR, {"start_pos": 1}) is not True:
        raise ExtractionError(
            f"g_decode 表达式 {G_DECODE_EXPR!r} 机器求值不符合 "
            "prefill=False/decode=True——guard 不可机器核验")
    if g.evidence_refs != ("ev_ex_spec_return",):
        raise ExtractionError(
            f"g_decode 证据引用 {g.evidence_refs!r} 应为 "
            "('ev_ex_spec_return',)（forward_spec 929-936 相位证据）")
    for e in graph.edges:
        if e.guard_ref != G_DECODE_ID:
            raise ExtractionError(
                f"末出口边 {e.stable_id} 相位 guard {e.guard_ref!r} 非 "
                f"{G_DECODE_ID!r}——forward_spec 933-934 prefill 在 stage "
                "loop 后 return，整条 head 链 decode-only；去 guard/换 "
                "guard 即 prefill 误启，具名拒绝")

    tensors = {t_.name: t_ for t_ in graph.tensors}
    nodes = {n.stable_id: n for n in graph.nodes}
    edges_by_id = {e.stable_id: e for e in graph.edges}

    # 必要节点/张量/边闭世界（先于具体端口核对，缺/多/重接都点名）
    missing_nodes = sorted(_EXPECTED_NODES - set(nodes))
    if missing_nodes:
        raise ExtractionError(
            f"末出口图缺少必要节点: {missing_nodes}——head 链断链具名拒绝")
    extra_nodes = sorted(set(nodes) - _EXPECTED_NODES)
    if extra_nodes:
        raise ExtractionError(
            f"末出口图出现必要链之外的节点: {extra_nodes}——第二 head "
            "通路/独立身份/多余节点（含未实现 Markov/Confidence 占位）"
            "具名拒绝")
    missing_tensors = sorted(_EXPECTED_TENSORS - set(tensors))
    if missing_tensors:
        raise ExtractionError(
            f"末出口图缺少必要张量: {missing_tensors}——断链具名拒绝")
    extra_tensors = sorted(set(tensors) - _EXPECTED_TENSORS)
    if extra_tensors:
        raise ExtractionError(
            f"末出口图出现必要链之外的张量: {extra_tensors}——共享 head "
            "复制身份/独立身份具名拒绝")
    for eid, src_n, src_p, dst_n, dst_p, tname in _EXPECTED_CHAIN:
        e = edges_by_id.get(eid)
        if e is None:
            raise ExtractionError(
                f"必要生产/消费链边 {eid} 缺失（期望 {src_n}.{src_p} → "
                f"{dst_n}.{dst_p}，张量 {tname}）——hc_head→norm→head 链"
                "断链具名拒绝")
        got = (e.src_node_id, e.src_port, e.dst_node_id, e.dst_port,
               e.tensor_id)
        want = (src_n, src_p, dst_n, dst_p, tensors[tname].stable_id)
        if got != want:
            raise ExtractionError(
                f"必要链边 {eid} 被一致重接：期望 {src_n}.{src_p} → "
                f"{dst_n}.{dst_p}（张量 {tname}），得到 "
                f"{got[0]}.{got[1]} → {got[2]}.{got[3]}（张量 {got[4]}）"
                "——源/目的端口被一致重接具名拒绝")
    extra_edges = sorted(set(edges_by_id) - {c[0] for c in _EXPECTED_CHAIN})
    if extra_edges:
        e = edges_by_id[extra_edges[0]]
        raise ExtractionError(
            f"末出口图出现必要链之外的边: {extra_edges}（{e.src_node_id}."
            f"{e.src_port} → {e.dst_node_id}.{e.dst_port}，张量 "
            f"{e.tensor_id}）——多余数据通路具名拒绝")

    # 必要节点算子/粒度/端口闭世界（fix01 ①）：引用检查只查节点 ID/
    # 连线，同 ID 下把 hc_head 门控 sigmoid_gate 换成 Sinkhorn 等
    # 关键算子引用合法——对每个必要节点固定核对 op_type/granularity
    # 及其必要生产消费端口（期望为可信常量 _EXPECTED_NODE_OPS，不从
    # 被检图生成），张量名经 tensors[name].stable_id 解析。
    for nid, (op, gr, ins, outs) in _EXPECTED_NODE_OPS.items():
        n = nodes[nid]
        if n.op_type != op or n.granularity != gr:
            raise ExtractionError(
                f"必要节点 {nid} 算子被改：op_type={n.op_type!r} "
                f"granularity={n.granularity!r}，期望 op_type={op!r} "
                f"granularity={gr!r}——同 ID 换关键算子（如 hc_head 门控 "
                "sigmoid_gate 非 Sinkhorn，model.py:714 sigmoid 加权门控）"
                "具名拒绝")
        got_ports = (tuple((p.name, p.tensor_id) for p in n.input_ports),
                     tuple((p.name, p.tensor_id) for p in n.output_ports))
        want_ports = (
            tuple((pn, tensors[tn].stable_id) for pn, tn in ins),
            tuple((pn, tensors[tn].stable_id) for pn, tn in outs))
        if got_ports != want_ports:
            raise ExtractionError(
                f"必要节点 {nid} 端口面被改：input="
                f"{got_ports[0]!r} output={got_ports[1]!r}，期望 "
                f"input={want_ports[0]!r} output={want_ports[1]!r}"
                "——必要生产消费端口不符具名拒绝")

    # dtype/shape 逐张量核对
    for name, dtype, syms in _EXPECTED_TENSOR_SHAPES:
        t_ = tensors[name]
        if t_.dtype != dtype:
            raise ExtractionError(
                f"张量 {name} dtype {t_.dtype!r} 应为 {dtype!r}")
        if tuple(t_.logical_layout) != syms:
            raise ExtractionError(
                f"张量 {name} 轴 {tuple(t_.logical_layout)} 应为 {syms}")

    # 消费轴可信值/表达式核对（fix01 ②）：named_axes 中本图消费的轴
    # （V/H/N_hc/N_draft/N_hcH/B/C1）必须与可信定义逐字段一致——
    # named_axes 与张量内嵌 axes 同步改值（V 129280→129281）时引用检查
    # 仍为 0（V2 只查两处互相同一致），故对照 rules AXIS_DEFS +
    # ModelValues 交叉派生的可信值核对，head_w [V,H] 等真实 shape 可验；
    # B 保持场景动态表达式语义（value=None 不伪造数值）。本图闭世界轴
    # 集恰为可信集（28 轴中 7 个被消费，多余轴不在闭世界内）。
    trusted = _trusted_exit_axes(values)
    axes_by_symbol = {a.symbol: a for a in graph.named_axes}
    consumed = set(trusted)
    if not consumed <= set(axes_by_symbol):
        raise ExtractionError(
            f"末出口图缺少本图消费轴: {sorted(consumed - set(axes_by_symbol))}"
            "——named_axes 断轴具名拒绝")
    for sym, want in trusted.items():
        got = axes_by_symbol[sym]
        fields = ("name", "meaning", "origin", "evidence_ref", "value",
                  "expression", "unknown_reason")
        got_f = tuple(getattr(got, f) for f in fields)
        want_f = tuple(getattr(want, f) for f in fields)
        if got_f != want_f:
            raise ExtractionError(
                f"named_axes 轴 {sym} 定义被改："
                + "；".join(f"{f}={g!r} 应为 {w!r}" for f, g, w in
                            zip(fields, got_f, want_f) if g != w)
                + f"——可信轴值（如 V=vocab_size=129280，run config）"
                "被改具名拒绝")
    # 张量内嵌 axes 与 named_axes 同名轴对象一致（引用检查 V2 已查
    # 定义相等）+ 布局符号与 axes 符号一致，且恰消费可信轴——同步改值
    # 在上面 named_axes 核对已拒，这里兜底张量 axes 的 symbol/布局面。
    for t_ in graph.tensors:
        for axis in t_.axes:
            if axis.symbol not in trusted:
                raise ExtractionError(
                    f"张量 {t_.name} 引用非本图消费闭世界轴 "
                    f"{axis.symbol!r}——轴闭世界具名拒绝")
        if tuple(t_.logical_layout) != tuple(a.symbol for a in t_.axes) \
                or tuple(t_.physical_layout) != tuple(a.symbol for a in t_.axes):
            raise ExtractionError(
                f"张量 {t_.name} 布局与 axes 符号不一致: "
                f"logical={tuple(t_.logical_layout)!r} "
                f"physical={tuple(t_.physical_layout)!r} "
                f"axes={[a.symbol for a in t_.axes]!r}")

    # 共享 head 单一身份：head_w 恰一、唯一消费边 exit.head.w、证据齐
    head_w = tensors["exit:head_w"]
    consumers = [e for e in graph.edges if e.tensor_id == head_w.stable_id
                 and e.dst_port == "w"]
    if len(consumers) != 1 or consumers[0].dst_node_id != "exit.head":
        raise ExtractionError(
            f"共享 head_w 消费边应恰为 exit.head_w.y → exit.head.w，得到 "
            f"{[(e.stable_id, e.src_node_id, e.dst_node_id) for e in consumers]}"
            "——第二消费边即共享 head 复制身份具名拒绝")
    head_node = nodes["exit.head"]
    if "ev_ex_parallel_head" not in head_node.evidence_refs:
        raise ExtractionError(
            "exit.head 缺 ev_ex_parallel_head 证据（full_logits=True "
            "语义锚定）")
    headw_node = nodes["exit.head_w"]
    if "ev_ex_shared_head_ref" not in headw_node.evidence_refs:
        raise ExtractionError(
            "exit.head_w 缺 ev_ex_shared_head_ref 证据（904 共享身份 "
            "非复制）")

    # 唯一输出：恰一 graph_output 节点消费 exit:logits
    outs = [n for n, nd in nodes.items() if nd.op_type == "graph_output"]
    if outs != ["exit.logits_out"]:
        raise ExtractionError(
            f"末出口 graph_output 节点应恰为 exit.logits_out，得到 {outs}"
            "——多余输出边界具名拒绝")
    ret_in = [e for e in graph.edges if e.dst_node_id == "exit.logits_out"]
    if len(ret_in) != 1 or ret_in[0].tensor_id != tensors["exit:logits"].stable_id \
            or ret_in[0].src_node_id != "exit.head":
        raise ExtractionError(
            "exit.logits_out 应恰消费 exit.head.y 的 exit:logits——输出"
            "来源被替换具名拒绝")
