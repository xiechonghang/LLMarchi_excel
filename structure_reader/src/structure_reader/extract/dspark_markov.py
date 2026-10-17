"""DeepSeek-V4-Flash-0731 DSpark Markov 递推与采样写回（工作包 B · 步骤 B3b）。

数据来源：models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py
（固定 SHA256 基线，:mod:`structure_reader.extract.evidence` 逐条核对原文）。
静态取证，不导入/不执行模型代码，不声称采样 token 数值可静态已知。

本轮范围（task_b3b_markov01.md——Markov 模块/测试/协调产物；head 链已由
head 定点补修01 交付且经 Codex 核验，本模块只读复用其 exit:logits 边界，
不重做 head；B3 整体仍 in_progress，本模块不是完整 B3/DSpark/V2 交付）：
1. forward_head 864-871 Markov 递推（K=N_draft=5 步真实依赖链）：
   外部 seed input_ids rank-1 [B]（与 B3a 同一边界）→ output_ids [B,K+1]
   第 0 列初始化（864-865 new_empty + col0 写）→ 外部初始 head logits
   [B,K,V]（graph_input，声明来源为已认可 head 图 exit:logits 边界，
   forward_head 863 full_logits=True 的返回）→ 受控 K=5 有序位置展开：
   第 i 步读已写 output_ids[:, i]（868）→ **同一** markov_w1 [V,Rm] 查表
   → embed [B,Rm]（802；ParallelEmbedding bf16）→ **同一** markov_w2
   [V,Rm] full_logits=True fp32 bias [B,V]（803；ParallelHead）→
   logits[:, i].add_(bias)（869）→ sample(logits[:, i], temperature)（871）
   → 写 output_ids[:, i+1] → 下一步读取。第 0/中间/末步（i=0/1..3/4）
   索引、五次 add/采样、六列（K+1）结果完整，五步真实依赖可联查。
2. 版本/身份（静态版本表达）：output_ids 与 logits 用显式 StateVersion
   链表示源码 in-place 更新（865/871 逐列写、869 逐步 add_）——这是
   源码 in-place 更新的静态版本表达，不执行数值。output_ids：v0
   （864-865 初始化后）→ v1..v5（871 第 i 步写第 i+1 列后，每步一版本）；
   logits：v0（863 head 输出、更新前）→ v1..v5（869 第 i 步 add bias 后）。
   每步消费边连接正确版本：第 i 步读 output_ids v{i}（含刚写的列）、
   消费 logits v{i}、产出 logits v{i+1} 与 output_ids v{i+1}；最终输出
   边界（936 返回）消费 v5 终版，不接 v0。
3. sample 语义（939-946）：temperature 依据 ModelArgs 默认 1（model.py:39）
   与 run config 无覆盖（ModelValues 无该字段，不伪称有）。
   source_checked 静态分支：temperature==0 → argmax（942-943）；非零路径
   max(temperature, 1e-5) 缩放（944）→ fp32 softmax（945）→ 指数噪声除
   Gumbel-max（946）→ argmax。默认 temperature=1 下活动路径是 Gumbel-max
   分支；输出 token 值未知，不静态猜 token、不称运行验证。
4. 参数身份/角色/相位：markov_w1 恰一 [V,Rm] bf16（ParallelEmbedding，
   torch.empty 无显式 dtype → 参考入口 950 set_default_dtype(bf16)）、
   markov_w2 恰一 [V,Rm] fp32（ParallelHead 729，checkpoint 存 bf16 分开
   记录）；五步复用同一身份，绑定 mtp.{last}.markov_head.markov_w1/
   markov_w2.weight（836 last 专有，B2 spec stage.markov_w1_w/markov_w2_w
   同源）。每步查表 embed（当前 token 的 embed）以有序输出边界提供给
   后续 Confidence（872 stack），本包不接 Confidence、不伪造 confidence
   输出。相位：整条 Markov 链与 head 同源 decode-only（forward_spec
   933-934 prefill return），全部边挂机器可检 g_decode。
5. 序列化映射（MARKOV_SCHEMA canonical JSON）：step/param/role/phase/
   boundary 映射公开给 Codex/B4 消费；guard 机器求值核对 + B2 角色重推导
   核对 + scope 边界（pending 显式列出 Confidence/整体拼接——删 pending
   即拒绝，防 overclaim）。

不做（本轮明确未实现，后续续包；不得把本模块当完整 B3b）：
Confidence（807-815/872-873，只交付 markov_embed 有序输出边界）、
forward_head 返回三元组整体拼接（外部 head/seed 来源标明，暂不整体
拼接）、B2 内部图 pending 关闭、B4 全量集成。图内无 Confidence 占位
节点——不把未实现链画成已实现。

可调用入口：
- build_dspark_markov_graph() → DSparkMarkovBundle（Markov 递推图 +
  证据注册表（B1/B2/head 同源 + 本包 ev_mk_*）+ ModelValues + B2 角色
  推导）
- verify_markov_graph(graph, values)（构建期已调用，测试/读回复用）
- build_markov_step_map(values) → step/param/role/phase/boundary 映射
- serialize_markov_step_map / deserialize_markov_step_map（canonical
  JSON 往返，读回全量核对 + guard 机器求值核对）

失败异常接口（以实际接口为准）：证据核对失败抛
structure_reader.extract.evidence.EvidenceError；图构造错误、
validate_graph_references 未通过、角色推导不一致、step 映射 JSON 结构
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
from structure_reader.extract.dspark_exit import (
    G_DECODE_EXPR,
    G_DECODE_ID,
    _exit_evidence_defs,
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
from structure_reader.graph_ir import Guard, NamedAxis, StateTransition
from structure_reader.model_registry import find_workspace_root
from structure_reader.rules.deepseek_v4_flash_0731 import (
    AXIS_DEFS,
    ModelValues,
    load_model_values,
)
from structure_reader.scenario_evaluator import evaluate_bool

__all__ = [
    "MARKOV_TEMPLATE_ID",
    "MARKOV_SCHEMA",
    "N_STEPS",
    "MARKOV_STEP_MAP_EXPECTED_TOP",
    "DSparkMarkovBundle",
    "verify_markov_graph",
    "build_dspark_markov_graph",
    "build_markov_step_map",
    "serialize_markov_step_map",
    "deserialize_markov_step_map",
]

MARKOV_TEMPLATE_ID = "deepseek_v4_flash_0731:dspark_markov"
MARKOV_SCHEMA = "dspark-markov-step-map/v1"

# K = N_draft（run config dspark_block_size=5）：forward_head 867
# range(self.block_size) 循环次数；六列结果 K+1。
N_STEPS = 5

# Markov 链相位 guard：与 head/B1 同文（deepseek_v4_dspark.py:769、
# dspark_exit.py:110）。forward_spec 933-934 prefill（start_pos==0）
# 在 stage loop 后 return——935 末出口 forward_head（含 Markov 循环
# 864-871）只在 decode（start_pos > 0）可达。
_G_DECODE_EVIDENCE = "ev_ex_spec_return"


# ---------------------------------------------------------------------------
# 证据目录：本包新增条目（ev_mk_*，仅 Markov 递推所需）。head 链条目
# （ev_ex_*）、stage/B1 条目复用同源注册（同 ID 同原文，重复注册即拒绝）。
# ---------------------------------------------------------------------------

_E = SourceEvidence


def _markov_evidence_defs() -> tuple[SourceEvidence, ...]:
    m = "model.py"
    return (
        _E("ev_mk_head_class", m, "DSparkMarkovHead", 795, 804,
           "class DSparkMarkovHead(nn.Module):\n"
           "    def __init__(self, vocab_size: int, dspark_markov_rank: int):\n"
           "        self.markov_w1 = ParallelEmbedding(vocab_size, dspark_markov_rank)\n"
           "        self.markov_w2 = ParallelHead(vocab_size, dspark_markov_rank)\n"
           "    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:\n"
           "        embed = self.markov_w1(token_ids)\n"
           "        logits = self.markov_w2(embed, full_logits=True)\n"
           "        return logits, embed",
           "Markov 头（836 last 专有实例化）：markov_w1 ParallelEmbedding "
           "[V,Rm]（798；torch.empty 无显式 dtype → 参考入口 950 "
           "set_default_dtype(bf16) → 存储 bf16）、markov_w2 ParallelHead "
           "[V,Rm]（799；ParallelHead weight fp32，729）。forward 802-804："
           "token_ids 查表 embed [B,Rm] → w2 full_logits=True fp32 logits "
           "[B,V]（bias 语义）→ 返回 (logits, embed) 二元组。每种参数恰一"
           "身份，五步复用同一 DSparkMarkovHead 实例", "source_checked"),
        _E("ev_mk_loop", m, "forward_head Markov loop", 864, 871,
           "output_ids = input_ids.new_empty(input_ids.size(0), self.block_size + 1)\n"
           "output_ids[:, 0] = input_ids\n"
           "markov_embeds = []\n"
           "for i in range(self.block_size):\n"
           "    logits_bias, markov_embed = self.markov_head(output_ids[:, i])\n"
           "    logits[:, i].add_(logits_bias)\n"
           "    markov_embeds.append(markov_embed)\n"
           "    output_ids[:, i + 1] = sample(logits[:, i], self.temperature)",
           "Markov 递推循环（K=self.block_size=N_draft=5 步）：864-865 "
           "output_ids [B,K+1] new_empty + 第 0 列 = seed input_ids 初始化；"
           "第 i 步（867-871）读已写 output_ids[:, i] → markov_head 查表 "
           "（w1 embed / w2 bias）→ logits[:, i].add_(bias)（869 in-place）"
           "→ markov_embeds.append（870）→ output_ids[:, i+1] = sample("
           "logits[:, i], temperature)（871）。第 i 步读取依赖第 i-1 步"
           "写回（i=0 读 seed 列）——五步真实依赖；in-place 更新的静态"
           "版本表达见图内显式 StateVersion 链，不执行数值", "source_checked"),
        _E("ev_mk_sample", m, "sample() Gumbel-max", 939, 946,
           "def sample(logits, temperature: float = 1.0):\n"
           "    if temperature == 0:\n"
           "        return logits.argmax(dim=-1)\n"
           "    logits = logits / max(temperature, 1e-5)\n"
           "    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)\n"
           "    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)",
           "sample（模块级函数，924/871 两个调用点）：temperature==0 静态"
           "分支 argmax（942-943）；非零路径 max(temperature, 1e-5) 缩放"
           "（944）→ fp32 softmax（945，dtype=torch.float32）→ exponential_"
           " 噪声除 + argmax Gumbel-max（946）。本输入 temperature 来源："
           "ModelArgs 默认 1（model.py:39，ev_ep_temp_default 同源），run "
           "config 无 temperature 字段无覆盖 → 活动路径 = Gumbel-max 分支；"
           "输出 token 值依赖运行时噪声，静态未知", "source_checked"),
        _E("ev_mk_temp_last", m, "self.temperature binding in forward_head", 826, 828,
           "self.block_size = args.dspark_block_size\n"
           "self.noise_token_id = args.dspark_noise_token_id\n"
           "self.temperature = args.temperature",
           "DSparkBlock.temperature = args.temperature（828）：forward_head "
           "871 传 self.temperature。ModelArgs.temperature 默认 1（model.py:"
           "39）；generate.py 66-68 args.temperature = temperature（CLI 默认 "
           "1.0）——实际配置无 run config 覆盖字段，ModelValues 无该字段，"
           "绑定表只记录来源不冒充字段", "source_checked"),
        _E("ev_mk_embed_stack", m, "markov_embed stack + confidence call", 872, 873,
           "markov_embed = torch.stack(markov_embeds, dim=1)\n"
           "confidence = self.confidence_head(x, markov_embed)",
           "每步查表 embed（当前 token 的 embed，非生成 token 的 embed）"
           "逐位 append（870）→ 872 torch.stack(markov_embeds, dim=1) → "
           "[B,K,Rm] 有序输出边界，供后续 Confidence（873 confidence_head "
           "消费）。Confidence 本轮未实现——只交付 markov_embed 边界，"
           "图内不画 confidence 占位节点", "source_checked"),
        _E("ev_mk_boundary_premise", m, "Markov graph external boundary premise", None, None,
           None,
           "Markov 图外部边界来源前提",
           "derived",
           derivation="外部 seed input_ids rank-1 [B]（与 B3a 首入口同一"
                      "边界：forward_spec 935 的 input_ids 实参 → "
                      "forward_head 864 new_empty 首列 865 = input_ids）；"
                      "外部初始 head logits [B,K,V] 来自已认可 head 图"
                      "（dspark_exit exit:logits 边界，forward_head 863 "
                      "full_logits=True 输出）——本图不复制 head 链内部"
                      "节点，只声明端口来源。936 forward_head 返回 "
                      "(output_ids, logits, confidence)：本图展开到 "
                      "output_ids 终版/logits 终版/markov_embed 有序边界，"
                      "confidence 未实现不画。两外部端口暂不整体拼接"
                      "（拼接属后续续包）"),
    )


# ---------------------------------------------------------------------------
# 轴
# ---------------------------------------------------------------------------


def _c1_axis() -> NamedAxis:
    """单元宽轴 C1（head 模块同源复制，防跨模块漂移；Markov 图不直接
    消费 C1，本函数不使用——head 图专有）。"""
    return NamedAxis(
        "C1", "unit_width",
        "单元宽（shape-(1,) 参数）",
        "推导 model.py:841", "model.py:841", value=1)


def _mk_axes(values: ModelValues) -> dict[str, NamedAxis]:
    """Markov 图轴：B1 模块轴同源复制 + V（rules 同源，head 同法）+ Rm
    （dspark_markov_rank，Markov 专有轴，rules 无定义——本模块可信常量
    与 ModelValues 交叉核对，见 _rm_axis）。"""
    axes = _attn_axes(values)
    by_symbol = {a.symbol: a for a in AXIS_DEFS}
    a = by_symbol["V"]
    v = values.vocab_size
    if a.value != v:
        raise ExtractionError(
            f"可信轴来源漂移：rules 轴 V value={a.value!r} 与 "
            f"ModelValues.vocab_size={v!r} 不一致——校验基准不可信，"
            "拒绝构建")
    axes["V"] = NamedAxis(
        symbol=a.symbol, name=a.name, meaning=a.meaning,
        origin=a.origin, evidence_ref=a.evidence_ref,
        value=a.value, expression=a.expression,
        unknown_reason=a.unknown_reason,
    )
    axes["Rm"] = _rm_axis(values)
    # K+1 列宽派生轴（output_ids [B,K+1]，model.py:864 self.block_size + 1）。
    axes["N_draft_plus_1"] = NamedAxis(
        "N_draft_plus_1", "dspark_block_size_plus_1",
        "output_ids 列宽 K+1（草稿步数+种子列）",
        "推导 model.py:864", "ev_mk_loop",
        value=values.dspark_block_size + 1)
    return axes


def _rm_axis(values: ModelValues) -> NamedAxis:
    """Rm（dspark_markov_rank）Markov 专有轴。rules AXIS_DEFS 无此定义
    （B1/head 图不消费），与 deepseek_v4_dspark 模块内轴同法：本模块
    可信常量 + ModelValues 交叉核对（run_config.json:11）。"""
    v = values.dspark_markov_rank
    if v != 256:
        raise ExtractionError(
            f"可信轴来源漂移：Rm 本模块可信常量 256 与 "
            f"ModelValues.dspark_markov_rank={v!r} 不一致——校验基准"
            "不可信，拒绝构建")
    return NamedAxis(
        "Rm", "dspark_markov_rank", "Markov 低秩秩", "run config "
        "dspark_markov_rank", "run_config.json:11", value=v)


def _trusted_mk_axes(values: ModelValues) -> dict[str, NamedAxis]:
    """本图消费轴的可信定义（不从被检图生成期望）。

    覆盖 _EXPECTED_TENSOR_SHAPES 出现的全部 symbol：V/Rm/H/N_draft 取
    rules AXIS_DEFS 同源定义并与 ModelValues 对应字段交叉核对（Rm 为
    本模块可信常量，rules 无定义）；B 取 rules 表达式轴同源（B 保持
    场景动态表达式语义，value=None）。"""
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
                       ("N_draft", "dspark_block_size")):
        a = by_symbol[sym]
        v = getattr(values, field)
        if a.value != v:
            raise ExtractionError(
                f"可信轴来源漂移：rules 轴 {sym} value={a.value!r} 与 "
                f"ModelValues.{field}={v!r} 不一致——校验基准不可信，"
                "拒绝构建")
        trusted[sym] = copy_def(a)
    trusted["Rm"] = _rm_axis(values)
    trusted["B"] = copy_def(by_symbol["B"])
    return trusted


# ---------------------------------------------------------------------------
# Markov 递推图
# ---------------------------------------------------------------------------

# 必要节点闭世界（seed/logits 外部端口 + output_ids 初始化 + K=5 步
# 每步 markov_head 三算子 + sample + 唯一输出边界）。节点 ID 内嵌真实
# 步索引 i（有序位置实例），i∈{0..4}。w1/w2 参数节点各恰一（共享身份，
# 五条消费边到各步 embed.w/bias.w）；每步 logits 更新拆 slice（读 v{i}
# 位置 i 的 [B,V]）→ bias_add → updated_slice → logits_write（写回产
# v{i+1}）——869 logits[:, i].add_ 的静态图表达。
_EXPECTED_NODES: frozenset[str] = frozenset(
    {"mk.seed_src", "mk.init_logits_src", "mk.output_ids_init",
     "mk.w1_param", "mk.w2_param", "mk.stack_embed",
     "mk.embed_out", "mk.logits_out", "mk.output_ids_out"}
    | {f"mk.step{i}.read_col" for i in range(N_STEPS)}
    | {f"mk.step{i}.embed" for i in range(N_STEPS)}
    | {f"mk.step{i}.bias" for i in range(N_STEPS)}
    | {f"mk.step{i}.slice" for i in range(N_STEPS)}
    | {f"mk.step{i}.add" for i in range(N_STEPS)}
    | {f"mk.step{i}.logits_write" for i in range(N_STEPS)}
    | {f"mk.step{i}.sample" for i in range(N_STEPS)}
    | {f"mk.step{i}.write_col" for i in range(N_STEPS)}
)

# 必要张量闭世界（多余张量 = 独立身份/复制）。mk:logits/mk:output_ids
# 是被 in-place 更新的存储张量（StateVersion 链 state_id 同名）。
_EXPECTED_TENSORS: frozenset[str] = frozenset(
    {"mk:seed_ids", "mk:init_logits", "mk:logits", "mk:output_ids",
     "mk:w1_w", "mk:w2_w", "mk:markov_embed"}
    | {f"mk:step{i}_col_read" for i in range(N_STEPS)}
    | {f"mk:step{i}_embed" for i in range(N_STEPS)}
    | {f"mk:step{i}_bias" for i in range(N_STEPS)}
    | {f"mk:step{i}_logits_slice" for i in range(N_STEPS)}
    | {f"mk:step{i}_logits_updated" for i in range(N_STEPS)}
    | {f"mk:step{i}_written" for i in range(N_STEPS)}
)

# 必要生产/消费链闭世界（边 ID → 源节点.端口 → 目的节点.端口 + 张量）。
# 版本链核对在 _EXPECTED_VERSIONS/_EXPECTED_TRANSITIONS/每步消费版本表。
# 每张张量的全部 data 边都在本表（闭世界：多余边具名拒绝）。
_EXPECTED_CHAIN: tuple[tuple[str, str, str, str, str], ...] = (
    # 外部边界（seed/logits 来源 premise）。第 0 步 slice_read（循环
    # 生成段）即外部 init_logits 的消费边（= logits 存储 v0 来源）。
    ("e_mk_seed_init", "mk.seed_src", "y", "mk.output_ids_init", "seed",
     "mk:seed_ids"),
    # 初始化（864-865）
    ("e_mk_init_col0", "mk.output_ids_init", "y", "mk.step0.read_col", "ids",
     "mk:output_ids"),
    # 共享参数身份：恰一 w1/w1_w、w2/w2_w，各 5 条消费边（每步 embed.w /
    # bias.w）——第二身份/复制张量即多余节点+张量+边，闭世界拒绝。
    *[(f"e_mk_s{i}_w1_embed", "mk.w1_param", "y",
       f"mk.step{i}.embed", "w", "mk:w1_w")
      for i in range(N_STEPS)],
    *[(f"e_mk_s{i}_w2_bias", "mk.w2_param", "y",
       f"mk.step{i}.bias", "w", "mk:w2_w")
      for i in range(N_STEPS)],
    # 每步链（867-871）：read_col → embed 查表（w1）→ bias（w2）→
    # slice（读更新前 logits 位置 i）→ bias_add → logits_write（写回
    # 产 v{i+1}）→ sample（读更新后 [B,V]）→ write_col
    *[(f"e_mk_s{i}_read_embed", f"mk.step{i}.read_col", "token",
       f"mk.step{i}.embed", "x", f"mk:step{i}_col_read")
      for i in range(N_STEPS)],
    *[(f"e_mk_s{i}_embed_bias", f"mk.step{i}.embed", "y",
       f"mk.step{i}.bias", "x", f"mk:step{i}_embed")
      for i in range(N_STEPS)],
    # 第 0 步 slice 消费外部 init_logits（= logits v0）；i≥1 消费前一步
    # logits_write 产出的 logits（= v{i}）——869 in-place 更新的版本链
    *[(f"e_mk_s{i}_slice_read",
       "mk.init_logits_src" if i == 0 else f"mk.step{i - 1}.logits_write",
       "y", f"mk.step{i}.slice", "logits",
       "mk:init_logits" if i == 0 else "mk:logits")
      for i in range(N_STEPS)],
    *[(f"e_mk_s{i}_slice_gather", f"mk.step{i}.slice", "slice",
       f"mk.step{i}.add", "x", f"mk:step{i}_logits_slice")
      for i in range(N_STEPS)],
    *[(f"e_mk_s{i}_bias_add", f"mk.step{i}.bias", "y",
       f"mk.step{i}.add", "bias", f"mk:step{i}_bias")
      for i in range(N_STEPS)],
    *[(f"e_mk_s{i}_add_write", f"mk.step{i}.add", "y",
       f"mk.step{i}.logits_write", "updated", f"mk:step{i}_logits_updated")
      for i in range(N_STEPS)],
    # sample 读 add 产出的更新后 [B,V]（871 sample(logits[:, i])——add_
    # 之后已含 bias），不接更新前 slice、不接 [B,K,V] 全张量
    *[(f"e_mk_s{i}_add_sample", f"mk.step{i}.add", "y",
       f"mk.step{i}.sample", "logits", f"mk:step{i}_logits_updated")
      for i in range(N_STEPS)],
    *[(f"e_mk_s{i}_sample_write", f"mk.step{i}.sample", "token",
       f"mk.step{i}.write_col", "token", f"mk:step{i}_written")
      for i in range(N_STEPS)],
    *[(f"e_mk_s{i}_write_ids", f"mk.step{i}.write_col", "y",
       f"mk.step{i + 1}.read_col" if i + 1 < N_STEPS else "mk.output_ids_out",
       "ids", "mk:output_ids")
      for i in range(N_STEPS)],
    # 每步 embed 有序汇聚 → 872 stack → markov_embed 输出边界
    *[(f"e_mk_s{i}_embed_stack", f"mk.step{i}.embed", "y",
       "mk.stack_embed", f"in{i}", f"mk:step{i}_embed")
      for i in range(N_STEPS)],
    ("e_mk_stack_out", "mk.stack_embed", "stacked", "mk.embed_out", "x",
     "mk:markov_embed"),
    # 最终 logits 输出（936 logits 终版 v5）：step4 logits_write 产出
    ("e_mk_logits_out", "mk.step4.logits_write", "y", "mk.logits_out", "x",
     "mk:logits"),
)


# 每步 logits 版本消费/产出表：第 i 步 slice 消费 v{i}（869 in-place
# 的静态版本表达）、add/sample 消费 v{i} 位置 i 更新后的 [B,V]、
# logits_write 产出 v{i+1}；logits_out 消费 v5 终版。
# (consumer_node, consumer_port, producer_node, producer_port,
#  version_sid, 期望张量名)。张量名明确：slice 消费存储边界（v0=
# 外部 mk:init_logits，i≥1=mk:logits 本体），sample 消费本步 updated
# [B,V]（v{i} 位置 i 更新后内容），logits_out 消费存储张量本体 v5。
_EXPECTED_STEP_VERSIONS: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("mk.step0.slice", "logits", "mk.init_logits_src", "y",
     "state:mk:logits:v0", "mk:init_logits"),
    ("mk.step1.slice", "logits", "mk.step0.logits_write", "y",
     "state:mk:logits:v1", "mk:logits"),
    ("mk.step2.slice", "logits", "mk.step1.logits_write", "y",
     "state:mk:logits:v2", "mk:logits"),
    ("mk.step3.slice", "logits", "mk.step2.logits_write", "y",
     "state:mk:logits:v3", "mk:logits"),
    ("mk.step4.slice", "logits", "mk.step3.logits_write", "y",
     "state:mk:logits:v4", "mk:logits"),
    ("mk.step0.sample", "logits", "mk.step0.add", "y",
     "state:mk:logits:v0", "mk:step0_logits_updated"),
    ("mk.step1.sample", "logits", "mk.step1.add", "y",
     "state:mk:logits:v1", "mk:step1_logits_updated"),
    ("mk.step2.sample", "logits", "mk.step2.add", "y",
     "state:mk:logits:v2", "mk:step2_logits_updated"),
    ("mk.step3.sample", "logits", "mk.step3.add", "y",
     "state:mk:logits:v3", "mk:step3_logits_updated"),
    ("mk.step4.sample", "logits", "mk.step4.add", "y",
     "state:mk:logits:v4", "mk:step4_logits_updated"),
    ("mk.logits_out", "x", "mk.step4.logits_write", "y",
     "state:mk:logits:v5", "mk:logits"),
)

# output_ids 版本消费表：第 0 步读 v0（init 后）、第 i 步（i≥1）读
# v{i}（第 i-1 步写后）；write_col 是对共享张量的版本更新表达（data 边
# 携带写入 token，版本推进由 transition 承载）；最终输出消费 v5。
# 版本→张量映射：单一存储张量 mk:output_ids。
_EXPECTED_IDS_VERSIONS: tuple[tuple[str, str, str], ...] = (
    ("mk.step0.read_col", "ids", "state:mk:output_ids:v0"),
    ("mk.step1.read_col", "ids", "state:mk:output_ids:v1"),
    ("mk.step2.read_col", "ids", "state:mk:output_ids:v2"),
    ("mk.step3.read_col", "ids", "state:mk:output_ids:v3"),
    ("mk.step4.read_col", "ids", "state:mk:output_ids:v4"),
    ("mk.output_ids_out", "ids", "state:mk:output_ids:v5"),
)


# 必要节点算子/粒度/端口闭世界（可信常量，与 _add_markov_flow 构建同文；
# 不从被检图生成期望）。端口 (名称, 张量名) 有序元组，张量名经
# tensors[name].stable_id 解析。引用检查只查节点 ID/连线，同 ID 下换
# 关键算子（sample 的 gumbel_max→argmax 偷换、add 的 bias_add→覆盖写）
# 引用合法——由此逐节点固定核对，具名拒绝。
def _expected_node_ops() -> dict[str, tuple[str, str,
                                       tuple[tuple[str, str], ...],
                                       tuple[tuple[str, str], ...]]]:
    ops: dict[str, tuple[str, str,
                         tuple[tuple[str, str], ...],
                         tuple[tuple[str, str], ...]]] = {
        "mk.seed_src": ("graph_input", "tensor_op", (),
                        (("y", "mk:seed_ids"),)),
        "mk.init_logits_src": ("graph_input", "tensor_op", (),
                               (("y", "mk:init_logits"),)),
        "mk.output_ids_init": ("tensor_init", "tensor_op",
                               (("seed", "mk:seed_ids"),),
                               (("y", "mk:output_ids"),)),
        # 共享参数单一身份节点（五步复用同一 markov_w1/w2，802/803）
        "mk.w1_param": ("markov_w1_param", "parameter_op", (),
                        (("y", "mk:w1_w"),)),
        "mk.w2_param": ("markov_w2_param", "parameter_op", (),
                        (("y", "mk:w2_w"),)),
    }
    for i in range(N_STEPS):
        ops[f"mk.step{i}.read_col"] = ("col_read", "index_op",
                                       (("ids", "mk:output_ids"),),
                                       (("token", f"mk:step{i}_col_read"),))
        ops[f"mk.step{i}.embed"] = ("embedding", "parameter_op",
                                    (("x", f"mk:step{i}_col_read"),
                                     ("w", "mk:w1_w")),
                                    (("y", f"mk:step{i}_embed"),))
        ops[f"mk.step{i}.bias"] = ("linear", "parameter_op",
                                   (("x", f"mk:step{i}_embed"),
                                    ("w", "mk:w2_w")),
                                   (("y", f"mk:step{i}_bias"),))
        # 869 logits[:, i] 的静态 slice（读 v{i} 位置 i 的 [B,V]）。
        # 第 0 步 logits 端口声明外部 init_logits（= logits 存储 v0 的
        # 来源边界）；i≥1 声明 logits 存储张量本体。
        ops[f"mk.step{i}.slice"] = ("logits_slice", "index_op",
                                    (("logits",
                                      "mk:init_logits" if i == 0
                                      else "mk:logits"),),
                                    (("slice", f"mk:step{i}_logits_slice"),))
        ops[f"mk.step{i}.add"] = ("bias_add", "tensor_op",
                                  (("x", f"mk:step{i}_logits_slice"),
                                   ("bias", f"mk:step{i}_bias")),
                                  (("y", f"mk:step{i}_logits_updated"),))
        # 写回位置（列 i）由 read_col 的列序与版本表固定（slice/add
        # 是纯 [B,V] 运算，位置语义挂 slice 索引表达）
        ops[f"mk.step{i}.logits_write"] = ("logits_write", "index_op",
                                           (("updated",
                                             f"mk:step{i}_logits_updated"),),
                                           (("y", "mk:logits"),))
        # sample(logits[:, i], temperature)（871）：恰一 logits 输入
        # （add 产出的更新后 [B,V]）；temperature 是标量绑定非张量输入
        ops[f"mk.step{i}.sample"] = ("gumbel_max_sample", "tensor_op",
                                     (("logits",
                                       f"mk:step{i}_logits_updated"),),
                                     (("token", f"mk:step{i}_written"),))
        ops[f"mk.step{i}.write_col"] = ("col_write", "index_op",
                                        (("token", f"mk:step{i}_written"),),
                                        (("y", "mk:output_ids"),))
    # 872 torch.stack(markov_embeds, dim=1)
    ops["mk.stack_embed"] = (
        "stack", "tensor_op",
        tuple((f"in{i}", f"mk:step{i}_embed") for i in range(N_STEPS)),
        (("stacked", "mk:markov_embed"),))
    ops["mk.embed_out"] = (
        "graph_output", "boundary_op",
        (("x", "mk:markov_embed"),),
        (("y", "mk:markov_embed"),))
    ops["mk.logits_out"] = ("graph_output", "boundary_op",
                            (("x", "mk:logits"),),
                            (("y", "mk:logits"),))
    ops["mk.output_ids_out"] = ("graph_output", "boundary_op",
                                (("ids", "mk:output_ids"),),
                                (("y", "mk:output_ids"),))
    return ops


# 张量 dtype/shape 闭世界核对表（name → dtype + 有序轴）。
_EXPECTED_TENSOR_SHAPES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("mk:seed_ids", "int64", ("B",)),
    ("mk:init_logits", "fp32", ("B", "N_draft", "V")),
    ("mk:logits", "fp32", ("B", "N_draft", "V")),
    ("mk:output_ids", "int64", ("B", "N_draft_plus_1")),
    ("mk:w1_w", "bf16", ("V", "Rm")),
    ("mk:w2_w", "fp32", ("V", "Rm")),
    ("mk:markov_embed", "bf16", ("B", "N_draft", "Rm")),
    *[("mk:step%d_col_read" % i, "int64", ("B",)) for i in range(N_STEPS)],
    *[("mk:step%d_embed" % i, "bf16", ("B", "Rm")) for i in range(N_STEPS)],
    *[("mk:step%d_bias" % i, "fp32", ("B", "V")) for i in range(N_STEPS)],
    *[("mk:step%d_logits_slice" % i, "fp32", ("B", "V"))
      for i in range(N_STEPS)],
    *[("mk:step%d_logits_updated" % i, "fp32", ("B", "V"))
      for i in range(N_STEPS)],
    *[("mk:step%d_written" % i, "int64", ("B",)) for i in range(N_STEPS)],
)


def _add_markov_tensors(b: _GraphBuilder, axes: dict[str, NamedAxis]
                        ) -> dict[str, str]:
    """Markov 链张量角色 → stable_id（前缀 mk:）。

    dtype 依据：seed/int64（B3a 同源 input_ids dtype）、init/logits
    fp32（head full_logits fp32）、w1 bf16（ParallelEmbedding torch.empty
    → set_default_dtype(bf16)，950）、w2 fp32（ParallelHead 729，checkpoint
    存 bf16 分开记录）、embed bf16（F.embedding(x, bf16 weight)）、bias
    fp32（w2 full_logits=True fp32 输出）、output_ids/col_read/written
    int64（input_ids.new_empty 保留 input_ids dtype，generate.py 35
    dtype=torch.long）、markov_embed bf16（每步 embed 原样 stack）。
    logits [B,N_draft,V] 单一存储张量（in-place 更新的版本承载者，
    StateVersion 链 state_id 同名）；slice/updated 每步 [B,V]（869
    logits[:, i] 位置切片）。K+1 列宽轴 N_draft_plus_1（rules 无静态
    定义，dspark_block_size+1 派生，model.py:864 self.block_size + 1）。"""
    t: dict[str, str] = {}

    def add(role: str, dtype: str, syms: tuple[str, ...],
            evidence: tuple[str, ...] = ()) -> str:
        tid = b.tensor(f"mk:{role}", dtype, syms, evidence)
        t[role] = tid
        return tid

    # 外部边界
    add("seed_ids", "int64", ("B",), ("ev_mk_boundary_premise",))
    add("init_logits", "fp32", ("B", "N_draft", "V"),
        ("ev_mk_boundary_premise",))
    # in-place 更新的共享存储张量（版本链 state_id 同名）
    add("logits", "fp32", ("B", "N_draft", "V"), ("ev_mk_loop",))
    add("output_ids", "int64", ("B", "N_draft_plus_1"), ("ev_mk_loop",))
    # 共享参数单一身份（w1 bf16 / w2 fp32，checkpoint bf16 分开记录）
    add("w1_w", "bf16", ("V", "Rm"), ("ev_mk_head_class",))
    add("w2_w", "fp32", ("V", "Rm"), ("ev_mk_head_class",))
    # 每步中间张量
    for i in range(N_STEPS):
        add(f"step{i}_col_read", "int64", ("B",), ("ev_mk_loop",))
        add(f"step{i}_embed", "bf16", ("B", "Rm"), ("ev_mk_head_class",))
        add(f"step{i}_bias", "fp32", ("B", "V"), ("ev_mk_head_class",))
        add(f"step{i}_logits_slice", "fp32", ("B", "V"), ("ev_mk_loop",))
        add(f"step{i}_logits_updated", "fp32", ("B", "V"), ("ev_mk_loop",))
        add(f"step{i}_written", "int64", ("B",), ("ev_mk_loop",))
    # 有序 embed 输出边界（872 stack [B,K,Rm]）
    add("markov_embed", "bf16", ("B", "N_draft", "Rm"), ("ev_mk_embed_stack",))
    return t


def _add_markov_flow(b: _GraphBuilder, t: dict[str, str]) -> None:
    """Markov 递推流（forward_head 864-871 + sample 939-946 静态分支）。

    相位规则：整条 Markov 链全部边挂可求值 g_decode（start_pos > 0）——
    forward_spec 933-934 prefill 在 stage loop 后 return，935 末出口
    forward_head（含 864-871 循环）prefill 不可达。参数身份节点静态
    存在，其到 decode-only 算子的消费边同样挂 g_decode（head/B1 惯例）。
    不通过删边实现屏蔽。

    版本表达：logits/output_ids 的 in-place 更新（869 add_ / 865+871
    逐列写）用显式 StateVersion 链表达——logits v0→v1..v5（每步一版本，
    logits_write 产出）、output_ids v0→v1..v5（每步写一列后一版本）。
    data 边携带张量内容；版本推进由 _add_markov_states 的 transition
    承载，边不挂 state_transition_ref（write_col/logits_write 是对共享
    存储的更新算子，非独立状态存储边——与 B1 cache 的 kind="state" 写后
    读不同，这里版本链在 check 里逐端口核对，不在边上伪造 state kind）。

    共享参数单一身份：w1/w2 参数节点各恰一（mk.w1_param/mk.w2_param）、
    单一张量 mk:w1_w/mk:w2_w，五条消费边连到各步 embed.w/bias.w——图
    结构本身表达同一身份复用（第二身份=多余节点+张量+边，闭世界拒绝）。"""
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

    # ===== 外部边界（forward_spec 935 input_ids 实参 + 863 head 输出）=====
    node("mk.seed_src", "graph_input", "tensor_op",
         (), (("y", t["seed_ids"]),), ("ev_mk_boundary_premise",))
    node("mk.init_logits_src", "graph_input", "tensor_op",
         (), (("y", t["init_logits"]),), ("ev_mk_boundary_premise",))

    # ===== output_ids 初始化（864-865）=====
    node("mk.output_ids_init", "tensor_init", "tensor_op",
         (("seed", t["seed_ids"]),), (("y", t["output_ids"]),),
         ("ev_mk_loop",))
    edge("e_mk_seed_init", ("mk.seed_src", "y"),
         ("mk.output_ids_init", "seed"), "seed_ids",
         ("ev_mk_boundary_premise",))

    # ===== 共享参数单一身份（802/803；markov_w1/w2 五步复用）=====
    node("mk.w1_param", "markov_w1_param", "parameter_op",
         (), (("y", t["w1_w"]),), ("ev_mk_head_class",))
    node("mk.w2_param", "markov_w2_param", "parameter_op",
         (), (("y", t["w2_w"]),), ("ev_mk_head_class",))
    for i in range(N_STEPS):
        edge(f"e_mk_s{i}_w1_embed", ("mk.w1_param", "y"),
             (f"mk.step{i}.embed", "w"), "w1_w", ("ev_mk_head_class",))
        edge(f"e_mk_s{i}_w2_bias", ("mk.w2_param", "y"),
             (f"mk.step{i}.bias", "w"), "w2_w", ("ev_mk_head_class",))

    # ===== K=5 步递推（867-871）=====
    for i in range(N_STEPS):
        # 第 i 步读已写列 output_ids[:, i]（i=0 读 seed 列，i≥1 读上一
        # 步 sample 写回列——版本核对见 _EXPECTED_IDS_VERSIONS）
        node(f"mk.step{i}.read_col", "col_read", "index_op",
             (("ids", t["output_ids"]),),
             (("token", t[f"step{i}_col_read"]),), ("ev_mk_loop",))
        # 同一 markov_w1 查表（802）
        node(f"mk.step{i}.embed", "embedding", "parameter_op",
             (("x", t[f"step{i}_col_read"]), ("w", t["w1_w"])),
             (("y", t[f"step{i}_embed"]),), ("ev_mk_head_class",))
        edge(f"e_mk_s{i}_read_embed", (f"mk.step{i}.read_col", "token"),
             (f"mk.step{i}.embed", "x"), f"step{i}_col_read",
             ("ev_mk_loop",))
        # 同一 markov_w2 full_logits=True fp32 bias（803）
        node(f"mk.step{i}.bias", "linear", "parameter_op",
             (("x", t[f"step{i}_embed"]), ("w", t["w2_w"])),
             (("y", t[f"step{i}_bias"]),), ("ev_mk_head_class",))
        edge(f"e_mk_s{i}_embed_bias", (f"mk.step{i}.embed", "y"),
             (f"mk.step{i}.bias", "x"), f"step{i}_embed",
             ("ev_mk_head_class",))
        # 869 logits[:, i].add_(bias)：slice 读 v{i} 位置 i 的 [B,V] →
        # bias_add → logits_write 写回产 v{i+1}。第 0 步 logits 端口
        # 声明外部 init_logits（= logits 存储 v0 来源边界）
        node(f"mk.step{i}.slice", "logits_slice", "index_op",
             (("logits", t["init_logits"] if i == 0
               else t["logits"]),),
             (("slice", t[f"step{i}_logits_slice"]),), ("ev_mk_loop",))
        node(f"mk.step{i}.add", "bias_add", "tensor_op",
             (("x", t[f"step{i}_logits_slice"]),
              ("bias", t[f"step{i}_bias"])),
             (("y", t[f"step{i}_logits_updated"]),), ("ev_mk_loop",))
        node(f"mk.step{i}.logits_write", "logits_write", "index_op",
             (("updated", t[f"step{i}_logits_updated"]),),
             (("y", t["logits"]),), ("ev_mk_loop",))
        # 第 0 步 slice 消费外部 init_logits（= logits v0）；i≥1 消费
        # 前一步 logits_write 产出（= v{i}）——869 in-place 版本链
        edge(f"e_mk_s{i}_slice_read",
             ("mk.init_logits_src", "y") if i == 0
             else (f"mk.step{i - 1}.logits_write", "y"),
             (f"mk.step{i}.slice", "logits"),
             "init_logits" if i == 0 else "logits", ("ev_mk_loop",))
        edge(f"e_mk_s{i}_slice_gather", (f"mk.step{i}.slice", "slice"),
             (f"mk.step{i}.add", "x"), f"step{i}_logits_slice",
             ("ev_mk_loop",))
        edge(f"e_mk_s{i}_bias_add", (f"mk.step{i}.bias", "y"),
             (f"mk.step{i}.add", "bias"), f"step{i}_bias",
             ("ev_mk_loop",))
        edge(f"e_mk_s{i}_add_write", (f"mk.step{i}.add", "y"),
             (f"mk.step{i}.logits_write", "updated"),
             f"step{i}_logits_updated", ("ev_mk_loop",))
        # sample(logits[:, i], temperature)（871）读 add 产出的更新后
        # [B,V]（869 add_ 之后已含 bias）；temperature 是标量绑定非
        # 张量输入（不进端口）
        node(f"mk.step{i}.sample", "gumbel_max_sample", "tensor_op",
             (("logits", t[f"step{i}_logits_updated"]),),
             (("token", t[f"step{i}_written"]),), ("ev_mk_sample",))
        edge(f"e_mk_s{i}_add_sample", (f"mk.step{i}.add", "y"),
             (f"mk.step{i}.sample", "logits"), f"step{i}_logits_updated",
             ("ev_mk_loop",))
        # output_ids[:, i + 1] = token（871 写回）
        node(f"mk.step{i}.write_col", "col_write", "index_op",
             (("token", t[f"step{i}_written"]),),
             (("y", t["output_ids"]),), ("ev_mk_loop",))
        edge(f"e_mk_s{i}_sample_write", (f"mk.step{i}.sample", "token"),
             (f"mk.step{i}.write_col", "token"), f"step{i}_written",
             ("ev_mk_loop",))

    # 跨步链：第 i 步 write_col → 第 i+1 步 read_col（同一 output_ids
    # 张量、相邻版本）；第 4 步 write_col → 最终输出边界。
    for i in range(N_STEPS):
        dst = (f"mk.step{i + 1}.read_col", "ids") if i + 1 < N_STEPS \
            else ("mk.output_ids_out", "ids")
        edge(f"e_mk_s{i}_write_ids", (f"mk.step{i}.write_col", "y"),
             dst, "output_ids", ("ev_mk_loop",))

    # 初始化 → 第 0 步读（865 写 col0 → 867/868 读 col0）
    edge("e_mk_init_col0", ("mk.output_ids_init", "y"),
         ("mk.step0.read_col", "ids"), "output_ids", ("ev_mk_loop",))

    # ===== 有序 embed 输出边界（870 append + 872 stack）=====
    node("mk.stack_embed", "stack", "tensor_op",
         tuple((f"in{i}", t[f"step{i}_embed"]) for i in range(N_STEPS)),
         (("stacked", t["markov_embed"]),), ("ev_mk_embed_stack",))
    for i in range(N_STEPS):
        edge(f"e_mk_s{i}_embed_stack", (f"mk.step{i}.embed", "y"),
             ("mk.stack_embed", f"in{i}"), f"step{i}_embed",
             ("ev_mk_embed_stack",))
    node("mk.embed_out", "graph_output", "boundary_op",
         (("x", t["markov_embed"]),), (("y", t["markov_embed"]),),
         ("ev_mk_embed_stack",))
    edge("e_mk_stack_out", ("mk.stack_embed", "stacked"),
         ("mk.embed_out", "x"), "markov_embed", ("ev_mk_embed_stack",))

    # ===== 最终输出边界（936 logits 终版 v5）=====
    node("mk.logits_out", "graph_output", "boundary_op",
         (("x", t["logits"]),), (("y", t["logits"]),), ("ev_mk_loop",))
    edge("e_mk_logits_out", ("mk.step4.logits_write", "y"),
         ("mk.logits_out", "x"), "logits", ("ev_mk_loop",))
    node("mk.output_ids_out", "graph_output", "boundary_op",
         (("ids", t["output_ids"]),), (("y", t["output_ids"]),),
         ("ev_mk_loop",))


# 版本链（静态版本表达，不执行数值）：logits v0..v5 / output_ids
# v0..v5。每版本 stable_id / state_id / label。logits v0 = 863 head 输出
# （更新前），v{i+1} = 第 i 步 869 add 后；output_ids v0 = 865 初始化后
# （col0=seed），v{i+1} = 第 i 步 871 写第 i+1 列后。
_LOGITS_VERSIONS = tuple(
    (f"state:mk:logits:v{i}", "mk:logits", f"v{i}") for i in range(N_STEPS + 1))
_IDS_VERSIONS = tuple(
    (f"state:mk:output_ids:v{i}", "mk:output_ids", f"v{i}")
    for i in range(N_STEPS + 1))

_EXPECTED_VERSIONS = _LOGITS_VERSIONS + _IDS_VERSIONS

# transition：第 i 步 add 前后（logits）、第 i 步写列前后（output_ids）。
# transition 是纯版本推进记录（静态表达源码 in-place 更新），不承载
# 相位条件——相位在消费/产出这些版本的每条 data 边上（g_decode）。
_EXPECTED_TRANSITIONS: tuple[StateTransition, ...] = tuple(
    [StateTransition(
        stable_id=f"tr_mk_logits_s{i}", state_id="mk:logits",
        before_version_id=f"state:mk:logits:v{i}",
        after_version_id=f"state:mk:logits:v{i + 1}",
        evidence_refs=("ev_mk_loop",))
     for i in range(N_STEPS)]
    + [StateTransition(
        stable_id=f"tr_mk_ids_s{i}", state_id="mk:output_ids",
        before_version_id=f"state:mk:output_ids:v{i}",
        after_version_id=f"state:mk:output_ids:v{i + 1}",
        evidence_refs=("ev_mk_loop",))
       for i in range(N_STEPS)])


def _add_markov_states(b: _GraphBuilder) -> None:
    for sid, state_id, label in _EXPECTED_VERSIONS:
        b._versions.append(
            _version(sid, state_id, label))
    for tr in _EXPECTED_TRANSITIONS:
        b._transitions.append(tr)


def _version(sid: str, state_id: str, label: str):
    from structure_reader.graph_ir import StateVersion
    return StateVersion(stable_id=sid, state_id=state_id,
                        version_label=label)


# ---------------------------------------------------------------------------
# 角色推导（B2 只读复用）与 step/param/role/phase/boundary 映射
# ---------------------------------------------------------------------------

# last stage Markov 专有模块（model.py 836 markov_head 实例化 + 867-871
# 循环；confidence_head 同分支创建但本轮未实现，不入 markov_modules）。
_MARKOV_MODULES = ("markov_head",)

_SCOPE_IMPLEMENTED = (
    "markov_head（w1 查表 embed / w2 full_logits bias）",
    "K=5 递推循环（output_ids 版本链 + logits 版本链）",
    "sample（temperature=1 Gumbel-max 活动分支，静态）",
    "markov_embed 有序输出边界（872 stack）",
)
_SCOPE_PENDING = (
    "confidence_head（807-815/873）",
    "forward_head 返回三元组整体拼接（外部 head/seed 端口暂不拼接）",
    "B4 集成",
)

_PHASE_PREFILL_REASON = (
    "forward_spec 933-934 prefill（start_pos==0）在 stage loop 后 return，"
    "935 末出口 forward_head（含 864-871 Markov 循环）prefill 不可达")
_PHASE_DECODE_REASON = (
    "start_pos > 0 才到达 forward_spec 935 forward_head——Markov 循环 "
    "864-871 decode-only")

# temperature 来源（ev_ep_temp_default 同源事实；ModelValues 无该字段）。
_TEMPERATURE_SOURCE = {
    "value": 1,
    "source": "ModelArgs 默认 temperature: float = 1（model.py:39）；"
              "run config 无 temperature 字段（无覆盖）；generate.py 66-68 "
              "CLI 默认 1.0。ModelValues 无该字段——绑定表只记录来源，"
              "不伪称 ModelValues 有此字段",
    "active_branch": "gumbel_max（temperature!=0 → 944 缩放 → 945 fp32 "
                     "softmax → 946 exponential 除 Gumbel-max → argmax）",
    "dormant_branch": "argmax_direct（temperature==0 → 942-943，本输入"
                      "不活动，静态分支仅记录）",
}

# boundaries/sample 段可信常量（builder 与 deserialize 单一事实源——文案
# 仅此一处定义，构造与读回共用同一构造函数，漂移即构造期可见；期望不从
# 被检 JSON 生成）。Codex 复现的四份篡改 JSON 全部落在此前未逐叶子核对
# 的字段（seed shape/final_logits dtype/embed note/gumbel 分支文本）。
_BOUNDARY_SEED_SOURCE = (
    "forward_spec 935 input_ids 实参（与 B3a 首入口同一边界）")
_BOUNDARY_INIT_LOGITS_SOURCE = (
    "已认可 head 图 dspark_exit exit:logits 边界（forward_head 863 "
    "full_logits=True）")
_BOUNDARY_EMBED_NOTE = (
    "每步当前 token 查表 embed 的有序 stack（872），供后续 Confidence；"
    "本包不接 Confidence、不伪造 confidence 输出")
_BOUNDARY_TOKEN_VALUES_UNKNOWN = "unknown（采样输出静态未知，不猜值）"
_SAMPLE_MODULE = "model.py:939-946 模块级 sample()"
_SAMPLE_BRANCH_ARGMAX_DIRECT = "temperature==0 → argmax（942-943）"
_SAMPLE_BRANCH_GUMBEL_MAX = (
    "max(temperature,1e-5) 缩放（944）→ fp32 softmax（945）→ "
    "exponential 除 Gumbel-max （946）→ argmax")
_SAMPLE_RUNTIME_VALUES = "输出 token 值依赖运行时噪声，静态未知"


def _trusted_markov_boundaries() -> dict:
    """boundaries 段可信期望（单一事实源：build_markov_step_map 构造与
    deserialize 读回共用，不从被检 JSON 生成）。全常量：shape/dtype/
    来源/版本/unknown 声明/markov_embed 当前 token embed note 逐字段。"""
    return {
        "external_inputs": {
            "seed_input_ids": {
                "shape": "[B]",
                "dtype": "int64",
                "source": _BOUNDARY_SEED_SOURCE,
            },
            "init_head_logits": {
                "shape": "[B,K,V]",
                "dtype": "fp32",
                "source": _BOUNDARY_INIT_LOGITS_SOURCE,
            },
        },
        "outputs": {
            "output_ids_final": {
                "shape": "[B,K+1]",
                "dtype": "int64",
                "version": f"state:mk:output_ids:v{N_STEPS}",
                "token_values": _BOUNDARY_TOKEN_VALUES_UNKNOWN,
            },
            "final_logits": {
                "shape": "[B,K,V]",
                "dtype": "fp32",
                "version": f"state:mk:logits:v{N_STEPS}",
            },
            "markov_embed": {
                "shape": "[B,K,Rm]",
                "dtype": "bf16",
                "note": _BOUNDARY_EMBED_NOTE,
            },
        },
    }


def _trusted_markov_sample() -> dict:
    """sample 段可信期望（单一事实源同上）：分支语义逐字段——argmax_direct
    仅 temperature==0 静态分支；gumbel_max 为非零温度 缩放→fp32 softmax→
    指数噪声除 Gumbel-max（"非零温度直接 argmax" 文本偷换在此具名拒绝）。"""
    return {
        "module": _SAMPLE_MODULE,
        "branches": {
            "argmax_direct": _SAMPLE_BRANCH_ARGMAX_DIRECT,
            "gumbel_max": _SAMPLE_BRANCH_GUMBEL_MAX,
        },
        "static_branch_source_checked": True,
        "runtime_values": _SAMPLE_RUNTIME_VALUES,
    }


def _check_closed_world(got, want, path: str) -> None:
    """递归闭世界核对：字典多键/缺键、列表长度、叶子值/类型不符均以
    叶子路径具名拒绝。布尔期望用 is 判别（0/1 冒充 True/False 拒绝）；
    int 期望拒绝 bool/float（False==0、2.0==2 类型篡改拒绝）；其余叶子
    精确相等。期望一律来自可信常量，不从被检 JSON 生成。"""
    if isinstance(want, dict):
        if not isinstance(got, dict):
            raise _reject(path, got, want)
        got_keys, want_keys = set(got), set(want)
        if got_keys != want_keys:
            raise ExtractionError(
                f"Markov step 映射 {path} 字段闭世界不符：多 "
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


def _derive_and_check_roles(values: ModelValues):
    """B2 derive_stage_bindings 只读复用 + Markov 角色一致性核对。"""
    roles = derive_stage_bindings(values.n_mtp_layers, values.n_layers)
    last = roles[-1]
    if not last.is_last:
        raise ExtractionError(
            f"B2 角色推导末 stage（{last.param_path}）is_last 不成立——"
            "Markov 角色推导与 model.py 836 分支不一致")
    for module in _MARKOV_MODULES:
        if module not in last.role_modules:
            raise ExtractionError(
                f"B2 角色推导末 stage 缺 markov 模块 {module!r}（"
                f"role_modules={last.role_modules}）——与 model.py 836 "
                "不一致")
    for r in roles[:-1]:
        if "markov_head" in r.role_modules:
            raise ExtractionError(
                f"非末 stage {r.param_path} 出现 markov_head 角色——836 "
                "分支只在 stage_id == n_mtp_layers-1 实例化 markov_head")
    return roles


def build_markov_step_map(values: ModelValues) -> dict:
    """step/param/role/phase/boundary 映射（canonical JSON 往返，
    deserialize 全量核对，公开给 Codex/B4 消费）。

    step 记录每步 i：读取列索引（output_ids 第 i 列）、写入列索引
    （第 i+1 列）、logits 位置 i、消费/产出版本、w1/w2 绑定路径与
    身份、embed 输出端口。K/N_draft 派生自 ModelValues（不从被检
    JSON 刷新）。"""
    roles = _derive_and_check_roles(values)
    last = roles[-1]
    sid = last.stage_id
    k = values.dspark_block_size
    if k != N_STEPS:
        raise ExtractionError(
            f"ModelValues.dspark_block_size={k} 与本图展开步数 {N_STEPS} "
            "不一致——受控 K=5 展开依据 run config，配置漂移须先核源")
    steps = []
    for i in range(N_STEPS):
        steps.append({
            "step": i,
            "read_col": i,
            "write_col": i + 1,
            "logits_pos": i,
            "logits_consume_version": f"state:mk:logits:v{i}",
            "logits_produce_version": f"state:mk:logits:v{i + 1}",
            "ids_consume_version": f"state:mk:output_ids:v{i}",
            "ids_produce_version": f"state:mk:output_ids:v{i + 1}",
            "w1": {
                "bind": f"mtp.{sid}.markov_head.markov_w1.weight",
                "identity": "mk_w1_shared",
                "dtype": "bf16",
            },
            "w2": {
                "bind": f"mtp.{sid}.markov_head.markov_w2.weight",
                "identity": "mk_w2_shared",
                "dtype": "fp32",
                "checkpoint_dtype": "bf16",
            },
            "embed_out_port": f"in{i}",
        })
    return {
        "schema": MARKOV_SCHEMA,
        "template_id": MARKOV_TEMPLATE_ID,
        "guard": {
            "id": G_DECODE_ID,
            "expr": G_DECODE_EXPR,
            "prefill_active": False,
            "decode_active": True,
        },
        "n_stages": len(roles),
        "markov_stage": {
            "stage_id": sid,
            "param_path": last.param_path,
            "markov_modules": list(_MARKOV_MODULES),
            "block_size": k,
        },
        "steps": steps,
        "params": {
            "markov_w1": {
                "bind": f"mtp.{sid}.markov_head.markov_w1.weight",
                "shape": [values.vocab_size, values.dspark_markov_rank],
                "dtype": "bf16",
                "op": "ParallelEmbedding",
            },
            "markov_w2": {
                "bind": f"mtp.{sid}.markov_head.markov_w2.weight",
                "shape": [values.vocab_size, values.dspark_markov_rank],
                "dtype": "fp32",
                "checkpoint_dtype": "bf16",
                "op": "ParallelHead (full_logits=True)",
            },
        },
        "temperature": dict(_TEMPERATURE_SOURCE),
        "phases": {
            "prefill": {"markov_chain_active": False,
                        "reason": _PHASE_PREFILL_REASON},
            "decode": {"markov_chain_active": True,
                       "reason": _PHASE_DECODE_REASON},
        },
        "boundaries": _trusted_markov_boundaries(),
        "sample": _trusted_markov_sample(),
        "scope": {
            "implemented": list(_SCOPE_IMPLEMENTED),
            "pending": list(_SCOPE_PENDING),
        },
    }


def serialize_markov_step_map(step_map: dict) -> str:
    """canonical JSON（sort_keys，紧凑分隔符，ensure_ascii=False）。"""
    return json.dumps(step_map, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


# step 映射顶层键闭世界（deserialize 单一事实源，__all__ 公开给测试）。
MARKOV_STEP_MAP_EXPECTED_TOP = frozenset(
    {"schema", "template_id", "guard", "n_stages", "markov_stage", "steps",
     "params", "temperature", "phases", "boundaries", "sample", "scope"})


def _reject(what: str, got, want) -> ExtractionError:
    return ExtractionError(
        f"Markov step 映射 {what} 不符：得到 {got!r}，期望 {want!r}"
        "——语义篡改具名拒绝")


def deserialize_markov_step_map(text: str, values: ModelValues) -> dict:
    """读回全量核对（递归闭世界）：顶层/guard/roles/steps/params/
    temperature/phases/boundaries/sample/scope 每个叶子字段对照可信期望
    （_trusted_markov_boundaries/_trusted_markov_sample 等共享构造函数，
    不从被检 JSON 生成）逐字段核对——shape/dtype/来源文案/embed note/
    sample 分支语义文本均精确相等，int/bool 精确类型（False==0、2.0==2
    类型篡改拒绝）；guard 表达式机器求值核对 + scope 边界（pending 缺失
    即拒绝）。任何不符以叶子路径具名拒绝。"""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"Markov step 映射 JSON 不合法: {exc}") from exc
    if not isinstance(doc, dict):
        raise ExtractionError("Markov step 映射必须是 JSON 对象")
    expected_top = set(MARKOV_STEP_MAP_EXPECTED_TOP)
    got_top = set(doc)
    if got_top != expected_top:
        raise ExtractionError(
            f"Markov step 映射顶层键闭世界不符：多 "
            f"{sorted(got_top - expected_top)} 缺 "
            f"{sorted(expected_top - got_top)}")
    if doc["schema"] != MARKOV_SCHEMA:
        raise _reject("schema", doc["schema"], MARKOV_SCHEMA)
    if doc["template_id"] != MARKOV_TEMPLATE_ID:
        raise _reject("template_id", doc["template_id"], MARKOV_TEMPLATE_ID)

    # guard：字段闭世界 + 表达式一致 + 机器求值活动性一致
    guard = doc["guard"]
    if not isinstance(guard, dict) or set(guard) != {
            "id", "expr", "prefill_active", "decode_active"}:
        raise ExtractionError(
            f"Markov step 映射 guard 字段闭世界不符: {guard!r}")
    if guard["id"] != G_DECODE_ID:
        raise _reject("guard.id", guard["id"], G_DECODE_ID)
    if guard["expr"] != G_DECODE_EXPR:
        raise _reject("guard.expr", guard["expr"], G_DECODE_EXPR)
    if guard["prefill_active"] is not False or guard["decode_active"] is not True:
        raise ExtractionError(
            f"Markov step 映射 guard 活动性被改: {guard!r}——prefill"
            "（start_pos==0）在 forward_spec 933-934 return，Markov 链"
            " prefill 不活动")
    if evaluate_bool(guard["expr"], {"start_pos": 0}) is not False:
        raise ExtractionError(
            f"guard 表达式 {guard['expr']!r} 在 start_pos=0 机器求值非 False"
            "——与 prefill_active=false 不一致")
    if evaluate_bool(guard["expr"], {"start_pos": 1}) is not True:
        raise ExtractionError(
            f"guard 表达式 {guard['expr']!r} 在 start_pos=1 机器求值非 True"
            "——与 decode_active=true 不一致")

    roles = _derive_and_check_roles(values)
    last = roles[-1]
    sid = last.stage_id

    # n_stages：类型 int（bool/float/str 拒绝）且等于 B2 派生 stage 数
    n_stages = doc["n_stages"]
    if isinstance(n_stages, bool) or not isinstance(n_stages, int):
        raise ExtractionError(
            f"Markov step 映射 n_stages 类型不符: {n_stages!r}"
            f"（{type(n_stages).__name__}）应为 int——类型篡改具名拒绝")
    if n_stages != len(roles):
        raise _reject("n_stages", n_stages, len(roles))

    # markov_stage：stage_id/param_path/markov_modules 与 B2 派生一致；
    # block_size 精确 int 类型核对（bool/float 拒绝）。
    want_ms = {
        "stage_id": sid,
        "param_path": last.param_path,
        "markov_modules": list(_MARKOV_MODULES),
        "block_size": values.dspark_block_size,
    }
    _check_closed_world(doc["markov_stage"], want_ms, "markov_stage")

    # steps：逐字段精确类型核对（K 步有序、读取列 i、写入列 i+1、logits
    # 位置 i、版本链、共享 w1/w2 绑定与身份）；step/read_col/write_col/
    # logits_pos 拒绝 bool/float（False==0、2.0==2 类型篡改具名拒绝）。
    steps = doc["steps"]
    if not isinstance(steps, list) or len(steps) != N_STEPS:
        raise ExtractionError(
            f"Markov step 映射 steps 数量 {len(steps) if isinstance(steps, list) else steps!r}"
            f" 应为 {N_STEPS}（dspark_block_size={values.dspark_block_size}）")
    for i, s in enumerate(steps):
        want = {
            "step": i,
            "read_col": i,
            "write_col": i + 1,
            "logits_pos": i,
            "logits_consume_version": f"state:mk:logits:v{i}",
            "logits_produce_version": f"state:mk:logits:v{i + 1}",
            "ids_consume_version": f"state:mk:output_ids:v{i}",
            "ids_produce_version": f"state:mk:output_ids:v{i + 1}",
            "w1": {
                "bind": f"mtp.{sid}.markov_head.markov_w1.weight",
                "identity": "mk_w1_shared",
                "dtype": "bf16",
            },
            "w2": {
                "bind": f"mtp.{sid}.markov_head.markov_w2.weight",
                "identity": "mk_w2_shared",
                "dtype": "fp32",
                "checkpoint_dtype": "bf16",
            },
            "embed_out_port": f"in{i}",
        }
        _check_closed_world(s, want, f"steps[{i}]")

    # params：形状/dtype/算子核对（V/Rm 数值派生自 ModelValues；shape
    # 列表逐元素精确类型核对）。
    want_params = {
        "markov_w1": {
            "bind": f"mtp.{sid}.markov_head.markov_w1.weight",
            "shape": [values.vocab_size, values.dspark_markov_rank],
            "dtype": "bf16",
            "op": "ParallelEmbedding",
        },
        "markov_w2": {
            "bind": f"mtp.{sid}.markov_head.markov_w2.weight",
            "shape": [values.vocab_size, values.dspark_markov_rank],
            "dtype": "fp32",
            "checkpoint_dtype": "bf16",
            "op": "ParallelHead (full_logits=True)",
        },
    }
    _check_closed_world(doc["params"], want_params, "params")

    # temperature：来源与活动分支核对（ModelValues 无该字段——任何
    # temperature 数值字段化尝试都改变键闭世界被拒）；value 精确 int
    # 类型核对（True/1.0 冒充 1 拒绝）。
    _check_closed_world(doc["temperature"], dict(_TEMPERATURE_SOURCE),
                        "temperature")

    # phases：与 guard 表达式机器求值一致；活动布尔 is 判别（0/1 冒充
    # False/True 拒绝）。
    _check_closed_world(doc["phases"], {
        "prefill": {"markov_chain_active": False,
                    "reason": _PHASE_PREFILL_REASON},
        "decode": {"markov_chain_active": True,
                   "reason": _PHASE_DECODE_REASON},
    }, "phases")

    # boundaries：全量递归闭世界核对——每个 boundary 的每个叶子字段
    # （shape/dtype/source/note/version/token_values）对照共享可信构造
    # _trusted_markov_boundaries()（不从被检 JSON 生成）：seed rank2
    # （[B]→[B,1]）、final_logits dtype（fp32→bf16）、markov_embed note
    # 反转（Confidence 输入来源反转）等逐字段篡改均具名拒绝。
    bnd = doc["boundaries"]
    want_bnd = _trusted_markov_boundaries()
    _check_closed_world(bnd, want_bnd, "boundaries")

    # sample：全量递归闭世界核对（module/branches 语义文本逐字段/
    # source_checked/runtime_values）——"gumbel_max 分支描述被换成非零
    # 温度直接 argmax"（跳过 softmax/噪声）在此具名拒绝；分支名集合正确
    # 但文本偷换不再放行。
    sample = doc["sample"]
    _check_closed_world(sample, _trusted_markov_sample(), "sample")

    # scope：边界闭世界——pending 缺失（如删掉 confidence_head）即拒绝，
    # 防把未实现链声称为已实现/范围外。
    scope = doc["scope"]
    want_scope = {"implemented": list(_SCOPE_IMPLEMENTED),
                  "pending": list(_SCOPE_PENDING)}
    if not isinstance(scope, dict) or set(scope) != {"implemented", "pending"}:
        raise ExtractionError(
            f"Markov step 映射 scope 字段闭世界不符: {scope!r}")
    for key, want_v in want_scope.items():
        if scope[key] != want_v:
            raise _reject(f"scope.{key}", scope[key], want_v)
    return doc


# ---------------------------------------------------------------------------
# 构建入口与图级语义核对
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DSparkMarkovBundle:
    """B3b Markov 包交付：Markov 递推图 + 证据注册表 + 模型数值 + B2
    角色推导。"""
    graph: SemanticGraph
    registry: EvidenceRegistry
    values: ModelValues
    roles: tuple


def build_dspark_markov_graph(
    workspace_root: Path | None = None,
    *,
    values: ModelValues | None = None,
    evidence_base: EvidenceBase | None = None,
) -> DSparkMarkovBundle:
    """构建 Markov 递推语义图与 step 映射（静态取证，不导入/不执行模型
    代码）。

    证据核对 → 同源注册（B1 Attention + B2 stage + head ev_ex_* + 本包
    ev_mk_*；B3a entry 证据不被本图引用，不注册）→ 图构建（含显式
    StateVersion/transition 版本链）→ 通用引用检查 → verify_markov_graph
    语义核对 → B2 角色推导一致性核对。任何失败具名拒绝、不给图
    （EvidenceError / ExtractionError）。"""
    root = Path(workspace_root) if workspace_root is not None \
        else find_workspace_root()
    values = values or load_model_values(root)
    evidence_base = evidence_base or load_evidence_base(root)
    registry = EvidenceRegistry(evidence_base, root)
    for ev in _attn_evidence_defs() + _stage_evidence_defs() \
            + _exit_evidence_defs() + _markov_evidence_defs():
        registry.register(ev)

    axes = _mk_axes(values)
    for axis in axes.values():
        if axis.evidence_ref.startswith("ev_") \
                and axis.evidence_ref not in registry:
            raise ExtractionError(
                f"轴 {axis.symbol} 引用了未注册证据: {axis.evidence_ref!r}")

    b = _GraphBuilder(axes)
    t = _add_markov_tensors(b, axes)
    _add_markov_flow(b, t)
    _add_markov_states(b)

    for ev_id in b.evidence_ids:
        if ev_id not in registry:
            raise ExtractionError(f"图引用了未注册证据: {ev_id!r}")

    guards = (Guard(G_DECODE_ID, G_DECODE_EXPR, (_G_DECODE_EVIDENCE,)),)
    graph = b.build(guards, MARKOV_TEMPLATE_ID)
    verify_markov_graph(graph, values)

    roles = _derive_and_check_roles(values)
    return DSparkMarkovBundle(graph=graph, registry=registry, values=values,
                              roles=roles)


def verify_markov_graph(graph: SemanticGraph, values: ModelValues) -> None:
    """Markov 递推图语义核对（构建期调用；测试/读回复用）。

    结构/身份/版本链/必要链/相位逐项核对，语义篡改（引用合法但语义错）
    具名拒绝；篡改负例应先证 validate_graph_references 为 0 再断言本函数
    拒绝（引用合法前提下的语义拒绝，非悬空引用兜底）：
    - 相位：唯一 guard g_decode（表达式原文一致、机器求值 start_pos=0
      为 False / 1 为 True）；任何边未挂 g_decode 即 prefill 误启具名
      拒绝（forward_spec 933-934 prefill 在 stage loop 后 return）；
    - 必要节点/张量/生产消费链闭世界：缺步、保 ID 一致重接、多余节点/
      张量/边（w1/w2 复制身份、第二 Markov 通路）——全部具名拒绝；
    - 必要节点算子/粒度/端口面闭世界：op_type/granularity/输入输出端口
      对 _expected_node_ops() 可信常量逐节点核对——同 ID 换关键算子
      （gumbel_max_sample→argmax_sample、bias_add→覆盖写）具名拒绝；
    - dtype/shape 逐张量核对（seed int64 [B]、w1 bf16 [V,Rm]、w2 fp32
      [V,Rm]、embed bf16 [B,Rm]、bias fp32 [B,V]、logits fp32 [B,K,V]、
      output_ids int64 [B,K+1]、markov_embed bf16 [B,K,Rm]）；
    - 消费轴可信值/表达式核对：named_axes 中本图消费轴（V/Rm/H/N_draft/
      B）对照 rules AXIS_DEFS + ModelValues 交叉派生的可信定义逐字段
      核对，同步改值（V 129280→129281）具名拒绝；B 保持场景动态表达式；
    - 共享参数单一身份：w1/w2 张量名恰一、每步消费边到本步节点但
      stable_id 同名（复用同一身份，复制身份=多余张量已拒）、w1 bf16
      w2 fp32 区分、绑定路径核对；
    - 版本链（静态版本表达）：恰 12 版本（logits/output_ids 各 v0..v5）、
      恰 10 transition（每步各一、方向 v{i}→v{i+1} 无环）；每步 add 的
      x 消费前一步版本、sample 消费 add 产出版本（871 读更新后 logits，
      不接 bias 前版本）；第 0 步 add 消费 init_logits（v0）；最终
      logits_out 消费 v5 终版（不接 v0 初始版本）；每步 read_col 消费
      本步前 output_ids 版本（第 i 步读 v{i}，跨步前驱）、output_ids_out
      消费 v5 终版；
    - 跨步前驱：第 i+1 步 read_col 的 ids 生产者是第 i 步 write_col
      （e_mk_s{i}_write_ids），缺边/一致重接具名拒绝；
    - 唯一输出：恰三 graph_output 节点（logits_out/output_ids_out/
      embed_out），embed_out 恰 K 个有序 embed 端口（ Confidence 未
      实现，不画占位）。"""
    if graph.template_id != MARKOV_TEMPLATE_ID:
        raise ExtractionError(
            f"Markov 图 template_id 不符: {graph.template_id!r}")

    # 相位：唯一 g_decode + 表达式一致 + 机器求值
    guards = {g.stable_id: g for g in graph.guards}
    if set(guards) != {G_DECODE_ID}:
        raise ExtractionError(
            f"Markov 图 guards 应恰为 [{G_DECODE_ID}]，得到 {sorted(guards)}"
            "——Markov 链 decode-only 单相位条件")
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
                f"Markov 边 {e.stable_id} 相位 guard {e.guard_ref!r} 非 "
                f"{G_DECODE_ID!r}——forward_spec 933-934 prefill 在 stage "
                "loop 后 return，整条 Markov 链 decode-only；去 guard/换 "
                "guard 即 prefill 误启，具名拒绝")

    tensors = {t_.name: t_ for t_ in graph.tensors}
    nodes = {n.stable_id: n for n in graph.nodes}
    edges_by_id = {e.stable_id: e for e in graph.edges}

    # 必要节点/张量/边闭世界（先于具体端口核对，缺/多/重接都点名）
    missing_nodes = sorted(_EXPECTED_NODES - set(nodes))
    if missing_nodes:
        raise ExtractionError(
            f"Markov 图缺少必要节点: {missing_nodes}——递推链断链具名拒绝")
    extra_nodes = sorted(set(nodes) - _EXPECTED_NODES)
    if extra_nodes:
        raise ExtractionError(
            f"Markov 图出现必要链之外的节点: {extra_nodes}——第二 Markov "
            "通路/独立身份/多余节点（含未实现 Confidence 占位）具名拒绝")
    missing_tensors = sorted(_EXPECTED_TENSORS - set(tensors))
    if missing_tensors:
        raise ExtractionError(
            f"Markov 图缺少必要张量: {missing_tensors}——断链具名拒绝")
    extra_tensors = sorted(set(tensors) - _EXPECTED_TENSORS)
    if extra_tensors:
        raise ExtractionError(
            f"Markov 图出现必要链之外的张量: {extra_tensors}——w1/w2 复制"
            "身份/独立身份具名拒绝")
    for eid, src_n, src_p, dst_n, dst_p, tname in _EXPECTED_CHAIN:
        e = edges_by_id.get(eid)
        if e is None:
            raise ExtractionError(
                f"必要生产/消费链边 {eid} 缺失（期望 {src_n}.{src_p} → "
                f"{dst_n}.{dst_p}，张量 {tname}）——Markov 递推链断链具名"
                "拒绝")
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
            f"Markov 图出现必要链之外的边: {extra_edges}（{e.src_node_id}."
            f"{e.src_port} → {e.dst_node_id}.{e.dst_port}，张量 "
            f"{e.tensor_id}）——多余数据通路具名拒绝")

    # 必要节点算子/粒度/端口闭世界：引用检查只查节点 ID/连线，同 ID 下
    # 把 sample 的 gumbel_max_sample 换成 argmax_sample 等关键算子引用
    # 合法——对每个必要节点固定核对 op_type/granularity 及其必要生产
    # 消费端口（期望为可信常量 _expected_node_ops()，不从被检图生成），
    # 张量名经 tensors[name].stable_id 解析。
    expected_ops = _expected_node_ops()
    for nid, (op, gr, ins, outs) in expected_ops.items():
        n = nodes[nid]
        if n.op_type != op or n.granularity != gr:
            raise ExtractionError(
                f"必要节点 {nid} 算子被改：op_type={n.op_type!r} "
                f"granularity={n.granularity!r}，期望 op_type={op!r} "
                f"granularity={gr!r}——同 ID 换关键算子（sample 的 Gumbel-"
                "max 语义 model.py:939-946，add 的 bias 加法 869 非覆盖"
                "写）具名拒绝")
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

    # 消费轴可信值/表达式核对：named_axes 中本图消费的轴（V/Rm/H/
    # N_draft/B/N_draft_plus_1）必须与可信定义逐字段一致——同步改值
    # （V 129280→129281）时引用检查仍为 0，故对照 rules AXIS_DEFS +
    # ModelValues 交叉派生的可信值核对。B 保持场景动态表达式语义。
    trusted = _trusted_mk_axes(values)
    axes_by_symbol = {a.symbol: a for a in graph.named_axes}
    core_consumed = set(trusted)
    if not core_consumed <= set(axes_by_symbol):
        raise ExtractionError(
            f"Markov 图缺少本图消费轴: "
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
                + f"——可信轴值（如 V=vocab_size=129280，Rm="
                  f"dspark_markov_rank={values.dspark_markov_rank}）被改"
                "具名拒绝")
    # N_draft_plus_1：本图派生轴（block_size+1，model.py:864）——派生
    # 数值与 ModelValues 交叉核对，非从被检图刷新。
    dgp1 = axes_by_symbol.get("N_draft_plus_1")
    if dgp1 is None or dgp1.value != values.dspark_block_size + 1:
        raise ExtractionError(
            f"派生轴 N_draft_plus_1 应为 value="
            f"{values.dspark_block_size + 1}（dspark_block_size+1，"
            "model.py:864），得到 "
            f"{dgp1.value if dgp1 is not None else None!r}——K+1 列宽"
            "被改具名拒绝")
    # 张量内嵌 axes 只引用本图消费闭世界轴 + 布局符号一致。
    allowed_axis_syms = core_consumed | {"N_draft_plus_1"}
    for t_ in graph.tensors:
        for axis in t_.axes:
            if axis.symbol not in allowed_axis_syms:
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

    # 共享参数单一身份：w1/w2 张量名恰一（单一存储身份，五条消费边复用
    # 同一 stable_id），每条消费边 dst 恰为各步 embed/bias 的 w 端口；
    # w1 bf16 / w2 fp32 区分（checkpoint 存储差异分开记录，不混淆）。
    for name, want_dtype, op_nodes in (
            ("mk:w1_w", "bf16", [f"mk.step{i}.embed" for i in range(N_STEPS)]),
            ("mk:w2_w", "fp32", [f"mk.step{i}.bias" for i in range(N_STEPS)])):
        if name not in tensors:
            raise ExtractionError(
                f"markov 参数张量 {name} 缺失——共享身份断链具名拒绝")
        if tensors[name].dtype != want_dtype:
            raise ExtractionError(
                f"张量 {name} dtype {tensors[name].dtype!r} 应为 "
                f"{want_dtype!r}——w1 bf16 / w2 fp32（checkpoint bf16 "
                "分开记录）区分具名拒绝")
        consumers = [e for e in graph.edges if e.tensor_id
                     == tensors[name].stable_id]
        if len(consumers) != N_STEPS \
                or [e.dst_node_id for e in consumers] != op_nodes \
                or any(e.dst_port != "w" for e in consumers):
            raise ExtractionError(
                f"{name} 消费边应恰 {N_STEPS} 条到 {op_nodes}.w，得到 "
                f"{[(e.stable_id, e.dst_node_id, e.dst_port) for e in consumers]}"
                "——缺步/重接/第二消费身份即共享参数复制或断链具名拒绝")

    # 版本链（静态版本表达）：恰 12 版本 + 恰 10 transition + 无环方向
    versions = {v.stable_id: v for v in graph.state_versions}
    want_versions = {sid: (state, label)
                     for sid, state, label in _EXPECTED_VERSIONS}
    if set(versions) != set(want_versions):
        raise ExtractionError(
            f"Markov 版本集不符：多 {sorted(set(versions) - set(want_versions))}"
            f" 缺 {sorted(set(want_versions) - set(versions))}——静态版本"
            "表达被增删具名拒绝")
    for sid, (state, label) in want_versions.items():
        v = versions[sid]
        if v.state_id != state or v.version_label != label:
            raise ExtractionError(
                f"版本 {sid} 归属被改：state={v.state_id!r} label="
                f"{v.version_label!r}，期望 state={state!r} label="
                f"{label!r}")
    transitions = {tr.stable_id: tr for tr in graph.state_transitions}
    want_transitions = {tr.stable_id: (tr.state_id, tr.before_version_id,
                                       tr.after_version_id)
                        for tr in _EXPECTED_TRANSITIONS}
    if set(transitions) != set(want_transitions):
        raise ExtractionError(
            f"Markov transition 集不符：多 "
            f"{sorted(set(transitions) - set(want_transitions))} 缺 "
            f"{sorted(set(want_transitions) - set(transitions))}"
            "——版本推进记录被增删具名拒绝")
    for tid, (state, before, after) in want_transitions.items():
        tr = transitions[tid]
        if (tr.state_id, tr.before_version_id, tr.after_version_id) != \
                (state, before, after):
            raise ExtractionError(
                f"transition {tid} 方向被改：{tr.before_version_id!r}→"
                f"{tr.after_version_id!r}，期望 {before!r}→{after!r}"
                "——in-place 更新方向（869 add_ / 871 逐列写）具名拒绝")

    # 每步消费版本核对：slice 消费 v{i}（第 0 步来自 init_logits_src，
    # i≥1 来自前一步 logits_write）/ sample 消费本步 add 产出的更新后
    # logits / logits_out 消费 v5 终版（不接 v0 初始版本）。
    _verify_step_versions(graph)


def _verify_step_versions(graph: SemanticGraph) -> None:
    """每步版本消费/产出核对（verify_markov_graph 的子步骤，独立函数
    便于测试直接调用）。

    对 _EXPECTED_STEP_VERSIONS（六元组含期望张量名）/ _EXPECTED_IDS_
    VERSIONS 逐条核对：期望消费端口 (node, port) 的入边必须存在、生产
    者/张量/版本归属正确。logits v0 = 外部边界张量 mk:init_logits（863
    head 输出 = logits 存储 v0），v1..v5 = 单一存储张量 mk:logits；
    sample 消费本步 updated [B,V]（v{i} 位置 i 更新后内容）；
    output_ids 全版本 = 单一存储张量 mk:output_ids。版本区分由生产者
    节点与 StateVersion/transition 链承载，不靠每步独立张量名。"""
    tensors = {t_.name: t_ for t_ in graph.tensors}
    checks = [
        (consumer, port, producer, prod_port, version,
         tensors[tname].stable_id)
        for consumer, port, producer, prod_port, version, tname
        in _EXPECTED_STEP_VERSIONS]
    for consumer, port, version in _EXPECTED_IDS_VERSIONS:
        producer = {
            "mk.step0.read_col": "mk.output_ids_init",
            "mk.output_ids_out": f"mk.step{N_STEPS - 1}.write_col",
        }.get(consumer)
        if producer is None:
            i = int(consumer.split(".")[1][4:])
            producer = f"mk.step{i - 1}.write_col"
        checks.append((consumer, port, producer, "y" if port == "ids"
                       else port, version,
                       tensors["mk:output_ids"].stable_id))
    for consumer, port, producer, prod_port, version, want_tid in checks:
        incoming = [e for e in graph.edges
                    if e.dst_node_id == consumer and e.dst_port == port]
        if len(incoming) != 1:
            raise ExtractionError(
                f"{consumer}.{port} 入边应恰一条，得到 "
                f"{[(e.stable_id, e.src_node_id) for e in incoming]}"
                f"（版本 {version}）——版本链断链/多通路具名拒绝")
        e = incoming[0]
        if e.tensor_id != want_tid:
            raise ExtractionError(
                f"{consumer}.{port} 消费版本不符：边 {e.stable_id} 张量 "
                f"{e.tensor_id!r}，期望 {want_tid!r}（版本 {version}）"
                "——第 i 步读取被一致重接到旧列/旧版本具名拒绝")
        if e.src_node_id != producer:
            raise ExtractionError(
                f"{consumer}.{port} 生产者不符：{e.src_node_id!r}，期望 "
                f"{producer!r}（版本 {version}）——跨步前驱被重接具名拒绝")
