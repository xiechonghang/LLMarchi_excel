"""DeepSeek-V4-Flash-0731 DSpark Confidence 末出口置信链（工作包 B · 步骤 B3b）。

数据来源：models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py
（固定 SHA256 基线，:mod:`structure_reader.extract.evidence` 逐条核对原文）。
静态取证，不导入/不执行模型代码，不声称 confidence 数值可静态已知。

本轮范围（task_b3b_confidence01.md——Confidence 模块/测试/协调产物；
head 链与 Markov 链已分别交付并经 Codex 核验，本模块只读复用两者的
外部输出边界做来源映射，不重做 head/Markov；B3 整体仍 in_progress，
本模块不是完整 B3/DSpark/V2 交付）：
1. 真实输入来源（外部边界 premise，来源映射机器可检）：
   hidden = 已认可 head 图 dspark_exit 的 exit:prenorm_x 边界
   （生产者 exit.hc_head.reduce.y，张量 exit:prenorm_x，bf16
   [B,K,H]——forward_head 862 hc_head 输出、863 last.norm 之前的 x；
   非 exit:norm_x、非普通 stage h_out）。
   markov_embed = 已认可 Markov 图 dspark_markov 的 mk:markov_embed
   边界（生产者 mk.stack_embed.stacked，张量 mk:markov_embed，bf16
   [B,K,Rm]——872 五步各自当前输入 token 查表 embed 的有序 stack；
   非本步新采样 token 查表替代）。
   本图不复制 head/Markov 内部节点，只声明端口来源；跨图整体集成
   属 B4，不冒称已完成。
2. 实质数据链（forward 813-815 逐算子落图）：两输入 → concat
   （hidden 在前、markov_embed 在后，814 torch.cat(dim=-1)）→
   [B,K,H+Rm] bf16 → 独立显式 float 转换节点（815 hidden.float()）
   → fp32 → Linear proj（单一末 stage 参数）→ [B,K,1] fp32 →
   squeeze(-1) → [B,K] fp32 confidence 输出。concat/float/proj/
   squeeze 均有实际节点、完整端口与连线；不是 sigmoid/softmax 概率，
   禁止虚构数值/置信区间。
3. 参数：末 stage mtp.{last}.confidence_head.proj.weight（837 仅
   stage_id == n_mtp_layers-1 实例化），真实 Linear(input_dim, 1,
   dtype=torch.float32)（811）权重 [1, H+Rm] fp32（810 源码注释：
   checkpoint 存 bf16，运行参数 fp32——两者分开记录）；Linear 默认
   bias=False（132），无 bias 参数，不虚构；不把转置写成
   [H+Rm, 1]。input_dim = args.dim + args.dspark_markov_rank（837）
   = H + Rm = 4352（本模块可信常量与 ModelValues 交叉核对）。
4. 角色/相位：B2 derive_stage_bindings 只读复用，仅 last（含
   single-stage 合成 first&last）拥有 confidence_head 角色；prefill
   在 forward_spec 933-934 stage loop 后 return，873 confidence_head
   调用只在 decode（start_pos > 0）可达——每条边挂机器可检
   g_decode，不通过删边实现屏蔽。K=N_draft=dspark_block_size（867
   range(self.block_size)），本包受控 K=5 展开。
5. 序列化映射（CONFIDENCE_SCHEMA canonical JSON）：输入来源映射
   （producer template/node/port/tensor）/链路算子/参数/相位/boundary
   /scope 公开给 Codex/B4 消费；deserialize 全量递归闭世界核对 +
   guard 机器求值核对（沿用 Markov 定点补修01 的逐叶子核对口径，
   不重演只核外层键放行叶子的漏洞）。

不做（本轮明确未实现，后续续包；不得把本模块当完整 B3b）：
forward_head 返回三元组整体拼接（output_ids/logits/confidence 同一
forward_head 874）、跨图整体 build（head+Markov+Confidence 端到端，
B4）、可视页 CSV、正式 V2 门禁。图内不画未实现的跨图拼接占位节点。

可调用入口：
- build_dspark_confidence_graph() → DSparkConfidenceBundle（Confidence
  图 + 证据注册表（B1/B2/head 同源 + 本包 ev_cf_*）+ ModelValues +
  B2 角色推导）
- verify_confidence_graph(graph, values)（构建期已调用，测试/读回复用）
- build_confidence_map(values) → 输入来源/链路/参数/相位/boundary 映射
- serialize_confidence_map / deserialize_confidence_map（canonical
  JSON 往返，读回全量核对 + guard 机器求值核对）

失败异常接口（以实际接口为准）：证据核对失败抛
structure_reader.extract.evidence.EvidenceError；图构造错误、
validate_graph_references 未通过、角色推导不一致、映射 JSON 结构
不合法或语义篡改抛
structure_reader.extract.deepseek_v4_reference.ExtractionError
（ValueError 子类）。均具名拒绝、失败不给图/不给映射。
"""

from __future__ import annotations

import copy
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
from structure_reader.extract.dspark_exit import (
    G_DECODE_EXPR,
    G_DECODE_ID,
    _exit_evidence_defs,
)
from structure_reader.extract.dspark_markov import (
    N_STEPS,
    _rm_axis,
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
    "CONFIDENCE_TEMPLATE_ID",
    "CONFIDENCE_SCHEMA",
    "CONFIDENCE_MAP_EXPECTED_TOP",
    "DSparkConfidenceBundle",
    "verify_confidence_graph",
    "build_dspark_confidence_graph",
    "build_confidence_map",
    "serialize_confidence_map",
    "deserialize_confidence_map",
]

CONFIDENCE_TEMPLATE_ID = "deepseek_v4_flash_0731:dspark_confidence"
CONFIDENCE_SCHEMA = "dspark-confidence-map/v1"

# head 链 decode-only guard 同文复用（dspark_exit.py:110-111）；
# forward_spec 933-934 prefill（start_pos==0）在 stage loop 后 return，
# 935 末出口 forward_head（含 873 confidence_head 调用）只在 decode 可达。
_G_DECODE_EVIDENCE = "ev_ex_spec_return"


# ---------------------------------------------------------------------------
# 证据目录：本包新增条目（ev_cf_*）。head ev_ex_* / stage/B1 条目复用
# 同源注册（同 ID 同原文，重复注册即拒绝）。
# ---------------------------------------------------------------------------

_E = SourceEvidence


def _confidence_evidence_defs() -> tuple[SourceEvidence, ...]:
    m = "model.py"
    return (
        _E("ev_cf_conf_head", m, "DSparkConfidenceHead", 807, 815,
           "class DSparkConfidenceHead(nn.Module):\n"
           "    def __init__(self, input_dim: int):\n"
           "        super().__init__()\n"
           "        # proj in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for fp32 confidence score.\n"
           "        self.proj = Linear(input_dim, 1, dtype=torch.float32)\n"
           "\n"
           "    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor):\n"
           "        hidden = torch.cat([hidden, markov_embed], dim=-1)\n"
           "        return self.proj(hidden.float()).squeeze(-1)",
           "Confidence 头（837 末 stage 专有实例化，input_dim = args.dim + "
           "args.dspark_markov_rank）：proj = Linear(input_dim, 1, "
           "dtype=torch.float32)（811）——fp32 运行参数，权重 [1, "
           "input_dim]（150 else 分支 [out,in]），810 源码注释 checkpoint "
           "存 bf16（分开记录）；Linear 默认 bias=False（132），无 bias "
           "参数。forward 813-815：torch.cat([hidden, markov_embed], "
           "dim=-1)（814，hidden 在前）→ hidden.float()（815 显式 fp32 "
           "转换）→ proj → squeeze(-1)（815）→ [B,K] fp32。输出是原始 "
           "confidence 分数，非 sigmoid/softmax 概率", "source_checked"),
        _E("ev_cf_linear", m, "Linear.__init__ (weight shape / no bias)", 132, 155,
           "    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):\n"
           "        super().__init__()\n"
           "        self.in_features = in_features\n"
           "        self.out_features = out_features\n"
           "        dtype = dtype or default_dtype\n"
           "        if dtype == torch.float4_e2m1fn_x2:\n"
           "            # FP4: weight is [out, in//2] in float4_e2m1fn_x2, logically [out, in] in fp4\n"
           "            # Scale is [out, in//32] in float8_e8m0fnu (1 scale per 32 fp4 elements along K)\n"
           "            self.weight = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2))\n"
           "            scale_out_features = out_features\n"
           "            scale_in_features = in_features // fp4_block_size\n"
           "            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))\n"
           "        elif dtype == torch.float8_e4m3fn:\n"
           "            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))\n"
           "            scale_out_features = (out_features + block_size - 1) // block_size\n"
           "            scale_in_features = (in_features + block_size - 1) // block_size\n"
           "            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))\n"
           "        else:\n"
           "            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))\n"
           "            self.register_parameter(\"scale\", None)\n"
           "        if bias:\n"
           "            self.bias = nn.Parameter(torch.empty(out_features))\n"
           "        else:\n"
           "            self.register_parameter(\"bias\", None)",
           "Linear 权重/偏置事实（proj 消费）：非 fp4/fp8 dtype 分支权重 "
           "[out_features, in_features]（149-150）→ proj [1, input_dim] "
           "（不转置成 [input_dim, 1]）；bias 默认 False（132）→ 154-155 "
           "register_parameter(\"bias\", None)，proj 无 bias 参数",
           "source_checked"),
        _E("ev_cf_call", m, "confidence_head call site", 872, 873,
           "        markov_embed = torch.stack(markov_embeds, dim=1)\n"
           "        confidence = self.confidence_head(x, markov_embed)",
           "调用点（forward_head 末尾）：872 markov_embed = torch.stack("
           "markov_embeds, dim=1)——五步各自当前输入 token 查表 embed 的"
           "有序 stack [B,K,Rm]；873 confidence = self.confidence_head(x, "
           "markov_embed)——x 是 862 hc_head 输出（863 last.norm 之前的 "
           "prenorm_x），不是 863 norm 之后的 norm_x。两实参即本图两个"
           "外部输入端口的来源", "source_checked"),
        _E("ev_cf_boundary_premise", m,
           "Confidence graph external boundary premise", None, None, None,
           "Confidence 图外部边界来源前提",
           "derived",
           derivation="hidden 外部端口 = 已认可 head 图 dspark_exit 的 "
                      "exit:prenorm_x 边界（生产者 exit.hc_head.reduce.y，"
                      "张量 exit:prenorm_x，bf16 [B,K,H]；forward_head 862 "
                      "hc_head 输出、863 last.norm 之前——非 exit:norm_x、"
                      "非普通 stage h_out）。markov_embed 外部端口 = 已认可"
                      " Markov 图 dspark_markov 的 mk:markov_embed 边界"
                      "（生产者 mk.stack_embed.stacked，张量 mk:markov_embed，"
                      "bf16 [B,K,Rm]；872 五步当前输入 token embed 有序 "
                      "stack——非本步新采样 token 查表替代）。本图不复制"
                      "head/Markov 内部节点，只声明端口来源；跨图整体集成"
                      "属 B4 未实现"),
    )


# ---------------------------------------------------------------------------
# 轴
# ---------------------------------------------------------------------------


def _cf_axes(values: ModelValues) -> dict[str, NamedAxis]:
    """Confidence 图轴：B1 模块轴同源复制 + Rm（Markov 模块同源）+
    H_plus_Rm/C1（Confidence 专有派生轴，rules 无定义——本模块可信
    常量与 ModelValues 交叉核对）。"""
    axes = _attn_axes(values)
    axes["Rm"] = _rm_axis(values)
    axes["H_plus_Rm"] = _h_plus_rm_axis(values)
    axes["C1"] = _c1_axis()
    return axes


def _h_plus_rm_axis(values: ModelValues) -> NamedAxis:
    """H_plus_Rm（confidence proj 输入宽，Linear input_dim）Confidence
    专有派生轴。rules AXIS_DEFS 无此定义：本模块可信常量 4352 与
    ModelValues（dim + dspark_markov_rank）交叉核对（model.py:837）。"""
    v = values.dim + values.dspark_markov_rank
    if v != 4352:
        raise ExtractionError(
            f"可信轴来源漂移：H_plus_Rm 本模块可信常量 4352 与 "
            f"ModelValues.dim({values.dim!r}) + dspark_markov_rank"
            f"({values.dspark_markov_rank!r})={v!r} 不一致——校验基准"
            "不可信，拒绝构建")
    return NamedAxis(
        "H_plus_Rm", "confidence_input_dim",
        "confidence proj 输入宽 H+Rm（Linear input_dim）",
        "推导 model.py:837", "ev_cf_conf_head", value=v)


def _c1_axis() -> NamedAxis:
    """C1 单元宽轴（confidence proj 输出维 out_features=1，811
    Linear(input_dim, 1)）。"""
    return NamedAxis(
        "C1", "unit_width",
        "单元宽（confidence proj 输出维 out_features=1）",
        "推导 model.py:811", "ev_cf_conf_head", value=1)


def _trusted_cf_axes(values: ModelValues) -> dict[str, NamedAxis]:
    """本图消费轴的可信定义（不从被检图生成期望）。

    覆盖 _EXPECTED_TENSOR_SHAPES 出现的全部 symbol：H/N_draft 取
    rules AXIS_DEFS 同源定义并与 ModelValues 对应字段交叉核对；Rm
    复用 Markov 模块同源（可信常量 256 + ModelValues 交叉核对）；
    H_plus_Rm/C1 为本模块可信常量派生；B 取 rules 表达式轴同源
    （B 保持场景动态表达式语义，value=None）。"""
    by_symbol = {a.symbol: a for a in AXIS_DEFS}

    def copy_def(a: object) -> NamedAxis:
        return NamedAxis(
            symbol=a.symbol, name=a.name, meaning=a.meaning,
            origin=a.origin, evidence_ref=a.evidence_ref,
            value=a.value, expression=a.expression,
            unknown_reason=a.unknown_reason,
        )

    trusted: dict[str, NamedAxis] = {}
    for sym, field in (("H", "dim"), ("N_draft", "dspark_block_size")):
        a = by_symbol[sym]
        v = getattr(values, field)
        if a.value != v:
            raise ExtractionError(
                f"可信轴来源漂移：rules 轴 {sym} value={a.value!r} 与 "
                f"ModelValues.{field}={v!r} 不一致——校验基准不可信，"
                "拒绝构建")
        trusted[sym] = copy_def(a)
    trusted["B"] = copy_def(by_symbol["B"])
    trusted["Rm"] = _rm_axis(values)
    trusted["H_plus_Rm"] = _h_plus_rm_axis(values)
    trusted["C1"] = _c1_axis()
    return trusted


# ---------------------------------------------------------------------------
# Confidence 前向链图（forward 813-815）
# ---------------------------------------------------------------------------

# 必要节点闭世界（两个外部输入端口 + concat/float/proj/squeeze 四算子
# + 单一参数节点 + 唯一输出边界）。节点/张量/边多余即独立身份/第二通路，
# 闭世界拒绝。
_EXPECTED_NODES: frozenset[str] = frozenset(
    {"cf.hidden_src", "cf.embed_src", "cf.concat", "cf.to_float",
     "cf.proj_w", "cf.proj", "cf.squeeze", "cf.confidence_out"})

_EXPECTED_TENSORS: frozenset[str] = frozenset(
    {"cf:hidden_in", "cf:markov_embed_in", "cf:concat", "cf:concat_f32",
     "cf:proj_w", "cf:proj_out", "cf:confidence"})

# 必要生产/消费链闭世界（边 ID → 源节点.端口 → 目的节点.端口 + 张量）。
# 每张张量的全部 data 边都在本表（闭世界：多余边具名拒绝）。
# concat 输入端口序（hidden 在前、embed 在后）由节点端口面闭世界固定。
_EXPECTED_CHAIN: tuple[tuple[str, str, str, str, str], ...] = (
    ("e_cf_hidden_in", "cf.hidden_src", "y", "cf.concat", "hidden",
     "cf:hidden_in"),
    ("e_cf_embed_in", "cf.embed_src", "y", "cf.concat", "embed",
     "cf:markov_embed_in"),
    ("e_cf_concat_float", "cf.concat", "y", "cf.to_float", "x",
     "cf:concat"),
    ("e_cf_float_proj", "cf.to_float", "y", "cf.proj", "x",
     "cf:concat_f32"),
    ("e_cf_w_proj", "cf.proj_w", "y", "cf.proj", "w", "cf:proj_w"),
    ("e_cf_proj_squeeze", "cf.proj", "y", "cf.squeeze", "x",
     "cf:proj_out"),
    ("e_cf_squeeze_out", "cf.squeeze", "y", "cf.confidence_out", "x",
     "cf:confidence"),
)

# 必要节点算子/粒度/端口闭世界（可信常量，与 _add_confidence_flow 构建
# 同文；不从被检图生成期望）。端口 (名称, 张量名) 有序元组——concat 的
# 输入端口序即 814 cat 的拼接序（hidden 在前、markov_embed 在后）。
# 引用检查只查节点 ID/连线，同 ID 下换关键算子（float_cast→identity
# 绕过显式 float、squeeze_last→其它维度、linear→matmul）引用合法——
# 由此逐节点固定核对，具名拒绝。
def _expected_node_ops() -> dict[str, tuple[str, str,
                                       tuple[tuple[str, str], ...],
                                       tuple[tuple[str, str], ...]]]:
    return {
        "cf.hidden_src": ("graph_input", "tensor_op", (),
                          (("y", "cf:hidden_in"),)),
        "cf.embed_src": ("graph_input", "tensor_op", (),
                         (("y", "cf:markov_embed_in"),)),
        "cf.concat": ("concat", "tensor_op",
                      (("hidden", "cf:hidden_in"),
                       ("embed", "cf:markov_embed_in")),
                      (("y", "cf:concat"),)),
        "cf.to_float": ("float_cast", "tensor_op",
                        (("x", "cf:concat"),),
                        (("y", "cf:concat_f32"),)),
        "cf.proj_w": ("confidence_proj_weight", "parameter_op", (),
                      (("y", "cf:proj_w"),)),
        "cf.proj": ("linear", "parameter_op",
                    (("x", "cf:concat_f32"), ("w", "cf:proj_w")),
                    (("y", "cf:proj_out"),)),
        "cf.squeeze": ("squeeze_last", "tensor_op",
                       (("x", "cf:proj_out"),),
                       (("y", "cf:confidence"),)),
        "cf.confidence_out": ("graph_output", "boundary_op",
                              (("x", "cf:confidence"),),
                              (("y", "cf:confidence"),)),
    }


# 张量 dtype/shape 闭世界核对表（name → dtype + 有序轴）。
# concat bf16（torch.cat 保持输入 dtype）、concat_f32 fp32（815
# .float() 独立显式转换）、proj_w fp32 [C1, H_plus_Rm]（811+150，
# [out,in]=[1,4352] 不转置；810 checkpoint 存 bf16 分开记录）、
# proj_out [B,K,C1]、confidence [B,K] fp32（squeeze(-1) 末维 1 移除）。
_EXPECTED_TENSOR_SHAPES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("cf:hidden_in", "bf16", ("B", "N_draft", "H")),
    ("cf:markov_embed_in", "bf16", ("B", "N_draft", "Rm")),
    ("cf:concat", "bf16", ("B", "N_draft", "H_plus_Rm")),
    ("cf:concat_f32", "fp32", ("B", "N_draft", "H_plus_Rm")),
    ("cf:proj_w", "fp32", ("C1", "H_plus_Rm")),
    ("cf:proj_out", "fp32", ("B", "N_draft", "C1")),
    ("cf:confidence", "fp32", ("B", "N_draft")),
)


def _add_confidence_tensors(b: _GraphBuilder, axes: dict[str, NamedAxis]
                            ) -> dict[str, str]:
    """Confidence 链张量角色 → stable_id（前缀 cf:）。dtype 依据见
    _EXPECTED_TENSOR_SHAPES 注释。"""
    t: dict[str, str] = {}

    def add(role: str, dtype: str, syms: tuple[str, ...],
            evidence: tuple[str, ...] = ()) -> str:
        tid = b.tensor(f"cf:{role}", dtype, syms, evidence)
        t[role] = tid
        return tid

    add("hidden_in", "bf16", ("B", "N_draft", "H"),
        ("ev_cf_boundary_premise",))
    add("markov_embed_in", "bf16", ("B", "N_draft", "Rm"),
        ("ev_cf_boundary_premise",))
    add("concat", "bf16", ("B", "N_draft", "H_plus_Rm"),
        ("ev_cf_conf_head",))
    add("concat_f32", "fp32", ("B", "N_draft", "H_plus_Rm"),
        ("ev_cf_conf_head",))
    add("proj_w", "fp32", ("C1", "H_plus_Rm"), ("ev_cf_conf_head",))
    add("proj_out", "fp32", ("B", "N_draft", "C1"), ("ev_cf_conf_head",))
    add("confidence", "fp32", ("B", "N_draft"), ("ev_cf_conf_head",))
    return t


def _add_confidence_flow(b: _GraphBuilder, t: dict[str, str]) -> None:
    """Confidence 前向流（forward 813-815 静态展开）。

    相位规则：整条链全部边挂可求值 g_decode（start_pos > 0）——
    forward_spec 933-934 prefill 在 stage loop 后 return，873
    confidence_head 调用 prefill 不可达。不通过删边实现屏蔽。

    外部边界：hidden_src/embed_src 是 graph_input 端口（premise 声明
    producer template/node/port/tensor 来源，见模块 docstring 与
    ev_cf_boundary_premise）；本图不复制 head/Markov 内部节点。"""
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

    # ===== 外部输入端口（873 两实参的来源边界）=====
    node("cf.hidden_src", "graph_input", "tensor_op",
         (), (("y", t["hidden_in"]),),
         ("ev_cf_boundary_premise", "ev_ex_hc_head"))
    node("cf.embed_src", "graph_input", "tensor_op",
         (), (("y", t["markov_embed_in"]),),
         ("ev_cf_boundary_premise", "ev_cf_call"))

    # ===== concat（814 torch.cat([hidden, markov_embed], dim=-1)，
    # hidden 在前、markov_embed 在后——端口序即拼接序）=====
    node("cf.concat", "concat", "tensor_op",
         (("hidden", t["hidden_in"]), ("embed", t["markov_embed_in"])),
         (("y", t["concat"]),), ("ev_cf_conf_head",))
    edge("e_cf_hidden_in", ("cf.hidden_src", "y"),
         ("cf.concat", "hidden"), "hidden_in",
         ("ev_cf_boundary_premise", "ev_ex_hc_head", "ev_cf_call"))
    edge("e_cf_embed_in", ("cf.embed_src", "y"),
         ("cf.concat", "embed"), "markov_embed_in",
         ("ev_cf_boundary_premise", "ev_cf_call"))

    # ===== 独立显式 float 转换（815 hidden.float()）=====
    node("cf.to_float", "float_cast", "tensor_op",
         (("x", t["concat"]),), (("y", t["concat_f32"]),),
         ("ev_cf_conf_head",))
    edge("e_cf_concat_float", ("cf.concat", "y"),
         ("cf.to_float", "x"), "concat", ("ev_cf_conf_head",))

    # ===== 单一末 stage 参数 + fp32 proj（811 构造 / 815 调用）=====
    node("cf.proj_w", "confidence_proj_weight", "parameter_op",
         (), (("y", t["proj_w"]),), ("ev_cf_conf_head", "ev_ex_exit_role"))
    node("cf.proj", "linear", "parameter_op",
         (("x", t["concat_f32"]), ("w", t["proj_w"])),
         (("y", t["proj_out"]),), ("ev_cf_conf_head",))
    edge("e_cf_float_proj", ("cf.to_float", "y"),
         ("cf.proj", "x"), "concat_f32", ("ev_cf_conf_head",))
    edge("e_cf_w_proj", ("cf.proj_w", "y"),
         ("cf.proj", "w"), "proj_w", ("ev_cf_conf_head",))

    # ===== squeeze(-1)（815 [B,K,1]→[B,K]）→ 唯一输出边界 =====
    node("cf.squeeze", "squeeze_last", "tensor_op",
         (("x", t["proj_out"]),), (("y", t["confidence"]),),
         ("ev_cf_conf_head",))
    edge("e_cf_proj_squeeze", ("cf.proj", "y"),
         ("cf.squeeze", "x"), "proj_out", ("ev_cf_conf_head",))
    node("cf.confidence_out", "graph_output", "boundary_op",
         (("x", t["confidence"]),), (("y", t["confidence"]),),
         ("ev_cf_conf_head",))
    edge("e_cf_squeeze_out", ("cf.squeeze", "y"),
         ("cf.confidence_out", "x"), "confidence", ("ev_cf_conf_head",))


# ---------------------------------------------------------------------------
# 角色推导（B2 只读复用）与映射
# ---------------------------------------------------------------------------

_SCOPE_IMPLEMENTED = (
    "confidence_head 前向链（814 concat → 815 float/proj/squeeze），"
    "concat/float/proj/squeeze 均为实际节点+完整端口+连线",
    "末 stage 单一参数 proj.weight [1,H+Rm] fp32（810 checkpoint 存 "
    "bf16 分开记录），无 bias（Linear 默认 bias=False）",
    "外部输入边界来源映射：exit:prenorm_x（head 图 exit.hc_head.reduce.y）"
    "+ mk:markov_embed（Markov 图 mk.stack_embed.stacked）",
    "decode-only 相位（forward_spec 933-934 prefill return，873 调用"
    "不可达），逐边机器可检 g_decode",
)
_SCOPE_PENDING = (
    "forward_head 返回三元组整体拼接（output_ids/logits/confidence "
    "同一 forward_head 874）",
    "跨图整体 build（head+Markov+Confidence 端到端，B4）",
    "可视页 CSV/正式 V2 独立门禁",
)

_PHASE_PREFILL_REASON = (
    "forward_spec 933-934 prefill（start_pos==0）在 stage loop 后 return，"
    "935 末出口 forward_head（含 873 confidence_head 调用）prefill 不可达")
_PHASE_DECODE_REASON = (
    "start_pos > 0 才到达 forward_spec 935 forward_head——confidence_head "
    "调用（873）decode-only")

_K_SEMANTICS = (
    "K=N_draft=dspark_block_size（forward_head 867 range(self.block_size) "
    "五步展开，K 与 N_draft 同义）")

_BOUNDARY_HIDDEN_SOURCE = (
    "hc_head 862 输出（709-716 加权汇聚后）、last.norm（863）之前的 x——"
    "非 exit:norm_x、非普通 stage h_out")
_BOUNDARY_EMBED_SOURCE = (
    "872 五步各自当前输入 token 查表 embed 的有序 stack——非本步新采样 "
    "token 查表替代")
_BOUNDARY_CONF_SEMANTICS = (
    "原始 confidence 分数（非 sigmoid/softmax 概率、非置信区间，"
    "无虚构数值）")
_BOUNDARY_CONF_RUNTIME_NOTE = (
    "runtime fp32；checkpoint proj 权重存 bf16（810 源码注释），两者"
    "分开记录")

_TRUSTED_INPUTS: dict = {
    "hidden": {
        "boundary": "exit:prenorm_x",
        "producer_template": "deepseek_v4_flash_0731:dspark_exit",
        "producer_node": "exit.hc_head.reduce",
        "producer_port": "y",
        "producer_tensor": "exit:prenorm_x",
        "shape": "[B,K,H]",
        "dtype": "bf16",
        "source": _BOUNDARY_HIDDEN_SOURCE,
    },
    "markov_embed": {
        "boundary": "mk:markov_embed",
        "producer_template": "deepseek_v4_flash_0731:dspark_markov",
        "producer_node": "mk.stack_embed",
        "producer_port": "stacked",
        "producer_tensor": "mk:markov_embed",
        "shape": "[B,K,Rm]",
        "dtype": "bf16",
        "source": _BOUNDARY_EMBED_SOURCE,
    },
}

_TRUSTED_CHAIN: dict = {
    "concat": {
        "op": "concat",
        "inputs": ["hidden", "markov_embed"],
        "order": "hidden 在前、markov_embed 在后（814 torch.cat "
                 "[hidden, markov_embed], dim=-1）",
        "input_shapes": ["[B,K,H]", "[B,K,Rm]"],
        "output_shape": "[B,K,H+Rm]",
        "dtype": "bf16",
    },
    "float_cast": {
        "op": "float_cast",
        "input": "concat",
        "output_shape": "[B,K,H+Rm]",
        "dtype": "fp32",
        "source": "815 hidden.float() 独立显式 float 转换节点——proj 只"
                  "消费 fp32 输入，不绕过、不合并进 concat/proj",
    },
    "proj": {
        "op": "linear",
        "input": "float_cast",
        "weight_param": "confidence_proj",
        "output_shape": "[B,K,1]",
        "dtype": "fp32",
        "source": "815 self.proj(...) 单一末 stage Linear（811 构造），"
                  "无 bias",
    },
    "squeeze": {
        "op": "squeeze_last",
        "input": "proj",
        "output_shape": "[B,K]",
        "dtype": "fp32",
        "source": "815 .squeeze(-1) 末维 1 移除，[B,K,1]→[B,K]",
    },
}

_TRUSTED_BOUNDARIES: dict = {
    "output_confidence": {
        "shape": "[B,K]",
        "dtype": "fp32",
        "semantics": _BOUNDARY_CONF_SEMANTICS,
        "runtime_dtype_note": _BOUNDARY_CONF_RUNTIME_NOTE,
    },
}

_TRUSTED_PHASES: dict = {
    "prefill": {"confidence_active": False,
                "reason": _PHASE_PREFILL_REASON},
    "decode": {"confidence_active": True,
               "reason": _PHASE_DECODE_REASON},
}


def _trusted_cf_params(values: ModelValues, sid: int) -> dict:
    """params 段可信期望（单一事实源：build_confidence_map 构造与
    deserialize 读回共用）。shape [1, H+Rm]（[out,in] 不转置），
    dtype fp32（runtime）/checkpoint_dtype bf16（810 注释）分开，
    bias None（Linear 默认 bias=False，不虚构）。"""
    return {
        "confidence_proj": {
            "bind": f"mtp.{sid}.confidence_head.proj.weight",
            "shape": [1, values.dim + values.dspark_markov_rank],
            "dtype": "fp32",
            "checkpoint_dtype": "bf16",
            "op": "Linear(input_dim, 1, dtype=torch.float32)",
            "bias": None,
        },
    }


def _trusted_cf_stage(values: ModelValues, roles: tuple) -> dict:
    """confidence_stage 段可信期望（单一事实源同上）。"""
    last = roles[-1]
    return {
        "stage_id": last.stage_id,
        "param_path": last.param_path,
        "confidence_modules": ["confidence_head"],
        "block_size": values.dspark_block_size,
        "k_semantics": _K_SEMANTICS,
    }


def _derive_and_check_roles(values: ModelValues):
    """B2 derive_stage_bindings 只读复用 + Confidence 角色一致性核对
    （仅 last / single-stage 合成拥有 confidence_head）。"""
    roles = derive_stage_bindings(values.n_mtp_layers, values.n_layers)
    last = roles[-1]
    if not last.is_last:
        raise ExtractionError(
            f"B2 角色推导末 stage（{last.param_path}）is_last 不成立——"
            "Confidence 角色推导与 model.py 837 分支不一致")
    if "confidence_head" not in last.role_modules:
        raise ExtractionError(
            f"B2 角色推导末 stage 缺 confidence_head 角色（"
            f"role_modules={last.role_modules}）——与 model.py 837 "
            "不一致")
    for r in roles[:-1]:
        if "confidence_head" in r.role_modules:
            raise ExtractionError(
                f"非末 stage {r.param_path} 出现 confidence_head 角色——"
                "834 分支只在 stage_id == n_mtp_layers-1 实例化 "
                "confidence_head")
    return roles


def _check_k(values: ModelValues) -> int:
    k = values.dspark_block_size
    if k != N_STEPS:
        raise ExtractionError(
            f"ModelValues.dspark_block_size={k} 与受控 K={N_STEPS} "
            "不一致——受控展开依据 run config，配置漂移须先核源")
    return k


def build_confidence_map(values: ModelValues) -> dict:
    """输入来源/链路/参数/相位/boundary 映射（canonical JSON 往返，
    deserialize 全量核对，公开给 Codex/B4 消费）。

    输入来源映射记录 producer template/node/port/tensor 四元组（外部
    边界 premise 同文）；K/N_draft 派生自 ModelValues（不从被检
    JSON 刷新）。返回值与内部可信常量递归隔离（嵌套 dict/list 每次深
    拷贝新容器），调用方对公开返回对象的任何变异不污染 deserialize
    的校验基准，也不影响其它已构建映射与后续 fresh build。"""
    roles = _derive_and_check_roles(values)
    sid = roles[-1].stage_id
    _check_k(values)
    # 公开返回值与内部可信常量递归隔离：_TRUSTED_* 是 deserialize 的
    # 校验基准，公开 builder 直接别名返回会让调用方通过正常 API 变异
    # 污染校验期望（inputs/chain/phases/boundaries 均含嵌套可变
    # dict/list），故每次深拷贝生成新容器。
    return {
        "schema": CONFIDENCE_SCHEMA,
        "template_id": CONFIDENCE_TEMPLATE_ID,
        "guard": {
            "id": G_DECODE_ID,
            "expr": G_DECODE_EXPR,
            "prefill_active": False,
            "decode_active": True,
        },
        "n_stages": len(roles),
        "confidence_stage": _trusted_cf_stage(values, roles),
        "inputs": copy.deepcopy(_TRUSTED_INPUTS),
        "chain": copy.deepcopy(_TRUSTED_CHAIN),
        "params": _trusted_cf_params(values, sid),
        "phases": copy.deepcopy(_TRUSTED_PHASES),
        "boundaries": copy.deepcopy(_TRUSTED_BOUNDARIES),
        "scope": {
            "implemented": list(_SCOPE_IMPLEMENTED),
            "pending": list(_SCOPE_PENDING),
        },
    }


def serialize_confidence_map(confidence_map: dict) -> str:
    """canonical JSON（sort_keys，紧凑分隔符，ensure_ascii=False）。"""
    return json.dumps(confidence_map, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


# 映射顶层键闭世界（deserialize 单一事实源，__all__ 公开给测试）。
CONFIDENCE_MAP_EXPECTED_TOP = frozenset(
    {"schema", "template_id", "guard", "n_stages", "confidence_stage",
     "inputs", "chain", "params", "phases", "boundaries", "scope"})


def _reject(what: str, got, want) -> ExtractionError:
    return ExtractionError(
        f"Confidence 映射 {what} 不符：得到 {got!r}，期望 {want!r}"
        "——语义篡改具名拒绝")


def _check_closed_world(got, want, path: str) -> None:
    """递归闭世界核对：字典多键/缺键、列表长度、叶子值/类型不符均以
    叶子路径具名拒绝。布尔期望用 is 判别（0/1 冒充 True/False 拒绝）；
    int 期望拒绝 bool/float（False==0、2.0==2 类型篡改拒绝）；None 期望
    用 is 判别（bias: None 不接受 False/0 冒充）；其余叶子精确相等。
    期望一律来自可信常量，不从被检 JSON 生成。"""
    if isinstance(want, dict):
        if not isinstance(got, dict):
            raise _reject(path, got, want)
        got_keys, want_keys = set(got), set(want)
        if got_keys != want_keys:
            raise ExtractionError(
                f"Confidence 映射 {path} 字段闭世界不符：多 "
                f"{sorted(got_keys - want_keys)} 缺 "
                f"{sorted(want_keys - got_keys)}")
        for key in sorted(want_keys):
            _check_closed_world(got[key], want[key], f"{path}.{key}")
        return
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            raise _reject(path, got, want)
        for idx, (g, w) in enumerate(zip(got, want)):
            _check_closed_world(g, w, f"{path}[{idx}]")
        return
    if isinstance(want, bool) or want is None:
        if got is not want:
            raise _reject(path, got, want)
        return
    if isinstance(want, int):
        if isinstance(got, bool) or not isinstance(got, int) or got != want:
            raise _reject(path, got, want)
        return
    if type(got) is not type(want) or got != want:
        raise _reject(path, got, want)


def deserialize_confidence_map(text: str, values: ModelValues) -> dict:
    """读回全量核对（递归闭世界）：顶层/guard/confidence_stage/inputs/
    chain/params/phases/boundaries/scope 每个叶子字段对照可信期望
    （_TRUSTED_* / _trusted_cf_* 共享构造函数，不从被检 JSON 生成）
    逐字段核对——边界来源 producer 四元组/shape/dtype/来源文案/链路
    算子与拼接序/参数形状与 dtype（runtime fp32 vs checkpoint bf16）/
    bias None/相位活动布尔均精确相等，int/bool 精确类型（False==0、
    2.0==2 类型篡改拒绝）；guard 表达式机器求值核对 + scope 边界
    （pending 缺失即拒绝）。任何不符以叶子路径具名拒绝。"""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"Confidence 映射 JSON 不合法: {exc}") from exc
    if not isinstance(doc, dict):
        raise ExtractionError("Confidence 映射必须是 JSON 对象")
    expected_top = set(CONFIDENCE_MAP_EXPECTED_TOP)
    got_top = set(doc)
    if got_top != expected_top:
        raise ExtractionError(
            f"Confidence 映射顶层键闭世界不符：多 "
            f"{sorted(got_top - expected_top)} 缺 "
            f"{sorted(expected_top - got_top)}")
    if doc["schema"] != CONFIDENCE_SCHEMA:
        raise _reject("schema", doc["schema"], CONFIDENCE_SCHEMA)
    if doc["template_id"] != CONFIDENCE_TEMPLATE_ID:
        raise _reject("template_id", doc["template_id"],
                      CONFIDENCE_TEMPLATE_ID)

    # guard：字段闭世界 + 表达式一致 + 机器求值活动性一致
    guard = doc["guard"]
    if not isinstance(guard, dict) or set(guard) != {
            "id", "expr", "prefill_active", "decode_active"}:
        raise ExtractionError(
            f"Confidence 映射 guard 字段闭世界不符: {guard!r}")
    if guard["id"] != G_DECODE_ID:
        raise _reject("guard.id", guard["id"], G_DECODE_ID)
    if guard["expr"] != G_DECODE_EXPR:
        raise _reject("guard.expr", guard["expr"], G_DECODE_EXPR)
    if guard["prefill_active"] is not False \
            or guard["decode_active"] is not True:
        raise ExtractionError(
            f"Confidence 映射 guard 活动性被改: {guard!r}——prefill"
            "（start_pos==0）在 forward_spec 933-934 return，Confidence"
            " 链 prefill 不活动")
    if evaluate_bool(guard["expr"], {"start_pos": 0}) is not False:
        raise ExtractionError(
            f"guard 表达式 {guard['expr']!r} 在 start_pos=0 机器求值非 "
            "False——与 prefill_active=false 不一致")
    if evaluate_bool(guard["expr"], {"start_pos": 1}) is not True:
        raise ExtractionError(
            f"guard 表达式 {guard['expr']!r} 在 start_pos=1 机器求值非 "
            "True——与 decode_active=true 不一致")

    roles = _derive_and_check_roles(values)
    sid = roles[-1].stage_id
    _check_k(values)

    # n_stages：类型 int（bool/float/str 拒绝）且等于 B2 派生 stage 数
    n_stages = doc["n_stages"]
    if isinstance(n_stages, bool) or not isinstance(n_stages, int):
        raise ExtractionError(
            f"Confidence 映射 n_stages 类型不符: {n_stages!r}"
            f"（{type(n_stages).__name__}）应为 int——类型篡改具名拒绝")
    if n_stages != len(roles):
        raise _reject("n_stages", n_stages, len(roles))

    _check_closed_world(doc["confidence_stage"],
                        _trusted_cf_stage(values, roles),
                        "confidence_stage")
    _check_closed_world(doc["inputs"], _TRUSTED_INPUTS, "inputs")
    _check_closed_world(doc["chain"], _TRUSTED_CHAIN, "chain")
    _check_closed_world(doc["params"], _trusted_cf_params(values, sid),
                        "params")
    _check_closed_world(doc["phases"], _TRUSTED_PHASES, "phases")
    _check_closed_world(doc["boundaries"], _TRUSTED_BOUNDARIES,
                        "boundaries")

    # scope：边界闭世界——pending 缺失（如删掉整体拼接/门禁）即拒绝，
    # 防把未实现链声称为已实现/范围外。
    scope = doc["scope"]
    want_scope = {"implemented": list(_SCOPE_IMPLEMENTED),
                  "pending": list(_SCOPE_PENDING)}
    if not isinstance(scope, dict) or set(scope) != {"implemented",
                                                     "pending"}:
        raise ExtractionError(
            f"Confidence 映射 scope 字段闭世界不符: {scope!r}")
    for key, want_v in want_scope.items():
        if scope[key] != want_v:
            raise _reject(f"scope.{key}", scope[key], want_v)
    return doc


# ---------------------------------------------------------------------------
# 构建入口与图级语义核对
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DSparkConfidenceBundle:
    """B3b Confidence 包交付：Confidence 前向链图 + 证据注册表 + 模型
    数值 + B2 角色推导。"""
    graph: SemanticGraph
    registry: EvidenceRegistry
    values: ModelValues
    roles: tuple


def build_dspark_confidence_graph(
    workspace_root: Path | None = None,
    *,
    values: ModelValues | None = None,
    evidence_base: EvidenceBase | None = None,
) -> DSparkConfidenceBundle:
    """构建 Confidence 前向链语义图与映射（静态取证，不导入/不执行
    模型代码）。

    证据核对 → 同源注册（B1 Attention + B2 stage + head ev_ex_* + 本包
    ev_cf_*）→ 图构建 → 通用引用检查 → verify_confidence_graph 语义
    核对 → B2 角色推导一致性核对。任何失败具名拒绝、不给图
    （EvidenceError / ExtractionError）。"""
    root = Path(workspace_root) if workspace_root is not None \
        else find_workspace_root()
    values = values or load_model_values(root)
    evidence_base = evidence_base or load_evidence_base(root)
    registry = EvidenceRegistry(evidence_base, root)
    for ev in _attn_evidence_defs() + _stage_evidence_defs() \
            + _exit_evidence_defs() + _confidence_evidence_defs():
        registry.register(ev)

    axes = _cf_axes(values)
    for axis in axes.values():
        if axis.evidence_ref.startswith("ev_") \
                and axis.evidence_ref not in registry:
            raise ExtractionError(
                f"轴 {axis.symbol} 引用了未注册证据: {axis.evidence_ref!r}")

    b = _GraphBuilder(axes)
    t = _add_confidence_tensors(b, axes)
    _add_confidence_flow(b, t)

    for ev_id in b.evidence_ids:
        if ev_id not in registry:
            raise ExtractionError(f"图引用了未注册证据: {ev_id!r}")

    guards = (Guard(G_DECODE_ID, G_DECODE_EXPR, (_G_DECODE_EVIDENCE,)),)
    graph = b.build(guards, CONFIDENCE_TEMPLATE_ID)
    verify_confidence_graph(graph, values)

    roles = _derive_and_check_roles(values)
    return DSparkConfidenceBundle(graph=graph, registry=registry,
                                  values=values, roles=roles)


def verify_confidence_graph(graph: SemanticGraph,
                            values: ModelValues) -> None:
    """Confidence 前向链图语义核对（构建期调用；测试/读回复用）。

    结构/身份/必要链/相位逐项核对，语义篡改（引用合法但语义错）具名
    拒绝；篡改负例应先证 validate_graph_references 为 0 再断言本函数
    拒绝（引用合法前提下的语义拒绝，非悬空引用兜底）：
    - 相位：唯一 guard g_decode（表达式原文一致、机器求值 start_pos=0
      为 False / 1 为 True）；任何边未挂 g_decode 即 prefill 误启具名
      拒绝（forward_spec 933-934 prefill 在 stage loop 后 return）；
    - 必要节点/张量/生产消费链闭世界：缺算子、保 ID 一致重接、多余
      节点/张量/边（第二参数身份、概率支路）——全部具名拒绝；每条
      必要链边核对 kind=data（如 e_cf_float_proj 是 concat_f32 fp32
      进 proj 的唯一真实数据边，改 control/select/state 即断开真实
      图链）且不携带 state_transition_ref（前向链无跨调用状态版本）；
    - 必要节点算子/粒度/端口面闭世界：op_type/granularity/输入输出
      端口对 _expected_node_ops() 可信常量逐节点核对——同 ID 换关键
      算子（float_cast→identity 绕过显式 float、squeeze_last→其它维、
      concat→add）或动端口序（hidden/embed 交换）具名拒绝；
    - dtype/shape 逐张量核对（hidden/embed 输入 bf16 [B,K,H]/[B,K,Rm]、
      concat bf16 [B,K,H+Rm]、concat_f32 fp32、proj_w fp32 [C1,H+Rm]、
      proj_out [B,K,C1]、confidence fp32 [B,K]）；
    - 消费轴可信值/表达式核对：named_axes 中本图消费轴（B/H/N_draft/
      Rm/H_plus_Rm/C1）对照 rules AXIS_DEFS + ModelValues 交叉派生的
      可信定义逐字段核对，同步改值（H 4096→4097）具名拒绝；B 保持
      场景动态表达式；
    - 参数单一身份：proj_w 张量名恰一、dtype fp32、恰一条消费边到
      cf.proj.w、布局 [C1, H_plus_Rm]（[out,in] 不转置）；无 bias
      张量/节点（闭世界拒绝多余身份）；checkpoint bf16 事实在映射
      分开记录，不与 runtime fp32 混淆；
    - float 显式性：concat.y 只被 to_float 消费、proj.x 只消费
      concat_f32（815 独立显式转换，不绕过）——由必要链闭世界逐边
      核对；
    - squeeze 秩：confidence 输出恰 [B, N_draft] 二维（squeeze(-1)
      末维 1 移除），不保留 [B,K,1]；
    - 唯一输出：恰一 graph_output 节点（cf.confidence_out）、恰两
      graph_input 节点（hidden_src/embed_src）。"""
    if graph.template_id != CONFIDENCE_TEMPLATE_ID:
        raise ExtractionError(
            f"Confidence 图 template_id 不符: {graph.template_id!r}")

    # 相位：唯一 g_decode + 表达式一致 + 机器求值
    guards = {g.stable_id: g for g in graph.guards}
    if set(guards) != {G_DECODE_ID}:
        raise ExtractionError(
            f"Confidence 图 guards 应恰为 [{G_DECODE_ID}]，得到 "
            f"{sorted(guards)}——Confidence 链 decode-only 单相位条件")
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
    if g.evidence_refs != (_G_DECODE_EVIDENCE,):
        raise ExtractionError(
            f"g_decode 证据引用 {g.evidence_refs!r} 应为 "
            f"({_G_DECODE_EVIDENCE!r},)（forward_spec 929-936 相位证据）")
    for e in graph.edges:
        if e.guard_ref != G_DECODE_ID:
            raise ExtractionError(
                f"Confidence 边 {e.stable_id} 相位 guard {e.guard_ref!r} "
                f"非 {G_DECODE_ID!r}——forward_spec 933-934 prefill 在 "
                "stage loop 后 return，整条 Confidence 链 decode-only；"
                "去 guard/换 guard 即 prefill 误启，具名拒绝")

    tensors = {t_.name: t_ for t_ in graph.tensors}
    nodes = {n.stable_id: n for n in graph.nodes}
    edges_by_id = {e.stable_id: e for e in graph.edges}

    # 必要节点/张量/边闭世界（先于具体端口核对，缺/多/重接都点名）
    missing_nodes = sorted(_EXPECTED_NODES - set(nodes))
    if missing_nodes:
        raise ExtractionError(
            f"Confidence 图缺少必要节点: {missing_nodes}——前向链断链"
            "具名拒绝")
    extra_nodes = sorted(set(nodes) - _EXPECTED_NODES)
    if extra_nodes:
        raise ExtractionError(
            f"Confidence 图出现必要链之外的节点: {extra_nodes}——第二"
            "通路/独立身份/多余节点（含未实现跨图拼接占位）具名拒绝")
    missing_tensors = sorted(_EXPECTED_TENSORS - set(tensors))
    if missing_tensors:
        raise ExtractionError(
            f"Confidence 图缺少必要张量: {missing_tensors}——断链具名拒绝")
    extra_tensors = sorted(set(tensors) - _EXPECTED_TENSORS)
    if extra_tensors:
        raise ExtractionError(
            f"Confidence 图出现必要链之外的张量: {extra_tensors}——参数"
            "复制身份/独立身份（含虚构 bias）具名拒绝")
    for eid, src_n, src_p, dst_n, dst_p, tname in _EXPECTED_CHAIN:
        e = edges_by_id.get(eid)
        if e is None:
            raise ExtractionError(
                f"必要生产/消费链边 {eid} 缺失（期望 {src_n}.{src_p} → "
                f"{dst_n}.{dst_p}，张量 {tname}）——Confidence 前向链"
                "断链具名拒绝")
        if e.kind != "data":
            raise ExtractionError(
                f"必要链边 {eid} kind={e.kind!r} 应为 'data'——该边表达 "
                f"{src_n}.{src_p} → {dst_n}.{dst_p} 的实际张量消费（如 "
                f"e_cf_float_proj 是 concat_f32 fp32 进 proj 的唯一真实"
                "数据边），改 control/select/state 即不再表达实际消费、"
                "断开真实图链，具名拒绝")
        if e.state_transition_ref is not None:
            raise ExtractionError(
                f"必要链边 {eid} 携带 state_transition_ref="
                f"{e.state_transition_ref!r}——Confidence 前向链无跨调用"
                "状态版本，state 身份不适用，具名拒绝")
        got = (e.src_node_id, e.src_port, e.dst_node_id, e.dst_port,
               e.tensor_id)
        want = (src_n, src_p, dst_n, dst_p, tensors[tname].stable_id)
        if got != want:
            raise ExtractionError(
                f"必要链边 {eid} 被一致重接：期望 {src_n}.{src_p} → "
                f"{dst_n}.{dst_p}（张量 {tname}），得到 "
                f"{got[0]}.{got[1]} → {got[2]}.{got[3]}（张量 {got[4]}）"
                "——源/目的端口被一致重接（输入源换 post-norm/错 embed/"
                "float 绕过）具名拒绝")
    extra_edges = sorted(set(edges_by_id) - {c[0] for c in _EXPECTED_CHAIN})
    if extra_edges:
        e = edges_by_id[extra_edges[0]]
        raise ExtractionError(
            f"Confidence 图出现必要链之外的边: {extra_edges}（"
            f"{e.src_node_id}.{e.src_port} → {e.dst_node_id}."
            f"{e.dst_port}，张量 {e.tensor_id}）——多余数据通路（第二"
            "参数消费/概率支路）具名拒绝")

    # 必要节点算子/粒度/端口闭世界：引用检查只查节点 ID/连线，同 ID 下
    # 把 float_cast 换 identity、squeeze_last 换其它维度等引用合法——
    # 对每个必要节点固定核对 op_type/granularity 及其必要生产消费端口
    # （期望为可信常量 _expected_node_ops()，不从被检图生成），张量名
    # 经 tensors[name].stable_id 解析。concat 输入端口序即 814 拼接序。
    expected_ops = _expected_node_ops()
    for nid, (op, gr, ins, outs) in expected_ops.items():
        n = nodes[nid]
        if n.op_type != op or n.granularity != gr:
            raise ExtractionError(
                f"必要节点 {nid} 算子被改：op_type={n.op_type!r} "
                f"granularity={n.granularity!r}，期望 op_type={op!r} "
                f"granularity={gr!r}——同 ID 换关键算子（814 concat/"
                "815 float/linear/squeeze 语义）具名拒绝")
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
                "——端口序/端口张量不符（含 concat hidden/embed 交换）"
                "具名拒绝")

    # dtype/shape 逐张量核对
    for name, dtype, syms in _EXPECTED_TENSOR_SHAPES:
        t_ = tensors[name]
        if t_.dtype != dtype:
            raise ExtractionError(
                f"张量 {name} dtype {t_.dtype!r} 应为 {dtype!r}——"
                "runtime fp32/checkpoint bf16 混淆或 dtype 篡改具名拒绝")
        if tuple(t_.logical_layout) != syms:
            raise ExtractionError(
                f"张量 {name} 轴 {tuple(t_.logical_layout)} 应为 {syms}"
                "——权重转置（[H+Rm,1]）/squeeze 秩错误具名拒绝")

    # 消费轴可信值/表达式核对：named_axes 中本图消费的轴（B/H/N_draft/
    # Rm/H_plus_Rm/C1）必须与可信定义逐字段一致——同步改值（H
    # 4096→4097）时引用检查仍为 0，故对照 rules AXIS_DEFS + ModelValues
    # 交叉派生的可信值核对。B 保持场景动态表达式语义。
    trusted = _trusted_cf_axes(values)
    axes_by_symbol = {a.symbol: a for a in graph.named_axes}
    core_consumed = set(trusted)
    if not core_consumed <= set(axes_by_symbol):
        raise ExtractionError(
            f"Confidence 图缺少本图消费轴: "
            f"{sorted(core_consumed - set(axes_by_symbol))}"
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
                + f"——可信轴值（如 H=dim={values.dim}，Rm="
                  f"dspark_markov_rank={values.dspark_markov_rank}，"
                  "H_plus_Rm=4352，C1=1）被改具名拒绝")
    # 张量内嵌 axes 只引用本图消费闭世界轴 + 布局符号一致。
    for t_ in graph.tensors:
        for axis in t_.axes:
            if axis.symbol not in core_consumed:
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

    # 参数单一身份：proj_w 张量名恰一（单一存储身份）、dtype fp32
    # （runtime；checkpoint bf16 在映射分开记录，不混淆）、布局
    # [C1, H_plus_Rm]（[out,in]，不转置）、恰一条消费边到 cf.proj.w；
    # 无 bias 张量/节点（闭世界已拒多余身份，此处再点名消费面）。
    if "cf:proj_w" not in tensors:
        raise ExtractionError(
            "confidence 参数张量 cf:proj_w 缺失——参数断链具名拒绝")
    consumers = [e for e in graph.edges
                 if e.tensor_id == tensors["cf:proj_w"].stable_id]
    if len(consumers) != 1 or consumers[0].dst_node_id != "cf.proj" \
            or consumers[0].dst_port != "w":
        raise ExtractionError(
            f"cf:proj_w 消费边应恰 1 条到 cf.proj.w，得到 "
            f"{[(e.stable_id, e.dst_node_id, e.dst_port) for e in consumers]}"
            "——参数复制/第二消费身份/断链具名拒绝")

    # 输入/输出端口面：恰两 graph_input（hidden_src/embed_src）、恰一
    # graph_output（confidence_out）——936 三元组中只交付 confidence，
    # 不画未实现拼接占位。
    gi = sorted(n for n, n_ in nodes.items()
                if n_.op_type == "graph_input")
    go = sorted(n for n, n_ in nodes.items()
                if n_.op_type == "graph_output")
    if gi != ["cf.embed_src", "cf.hidden_src"]:
        raise ExtractionError(
            f"Confidence 图 graph_input 节点应恰 [cf.embed_src, "
            f"cf.hidden_src]，得到 {gi}——外部边界被增删具名拒绝")
    if go != ["cf.confidence_out"]:
        raise ExtractionError(
            f"Confidence 图 graph_output 节点应恰 [cf.confidence_out]，"
            f"得到 {go}——唯一输出被增删具名拒绝")
