"""DeepSeek-V4-Flash-0731 声明式规则：轴字典、层分组、守卫目录、场景集。

本模块是纯声明 + 少量受控推导：
- 轴字典（AXIS_DEFS）：symbol → 含义/来源；具体值来自 run config（ dim 等），
  场景相关轴（B/T/P）保存表达式，不预设数值。
- 层分组（derive_layer_groups）：由 compress_ratios + n_hash_layers 程序化推导
  四类主干分组，并集必须等于 0..n_layers-1 且互不重叠——未知 ratio、未支持
  组合、DSpark 尾部异常与推导失败一律具名报错，不静默产出错误分组；
  template_status 只有真实提取完成后才可标 extracted。
- 守卫目录（GUARD_DEFS）：predicate_expr 一律使用有限表达式求值器白名单语法，
  与源码条件一一对应（evidence 指向 model.py 行号）。
- 场景（SCENARIOS）：每个场景给 B/T/P/start_pos 环境；其余常量来自 config。

数据来源（evidence 键）：
- "run_config.json"  = models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/config.json
- "root_config.json" = models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json
- "model.py"         = models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

from structure_reader.scenario_evaluator import ExpressionError, check_expression

__all__ = [
    "RulesError",
    "ModelValues",
    "AxisDef",
    "GuardDef",
    "LayerGroup",
    "ScenarioDef",
    "AXIS_DEFS",
    "GUARD_DEFS",
    "SCENARIOS",
    "load_model_values",
    "derive_layer_groups",
]

# 相对 reference_root 的源码键 → 工作区相对路径；证据解析共用。
SOURCE_FILES = {
    "model.py": "models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py",
    "kernel.py": "models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/kernel.py",
    "run_config.json": "models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/config.json",
    "root_config.json": "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json",
    "generate.py": "models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/generate.py",
}


class RulesError(ValueError):
    """规则数据或推导不合法（具名拒绝）。"""


# ---------------------------------------------------------------------------
# 轴字典
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisDef:
    """一条轴定义：symbol、英文名、中文含义、来源说明与证据键。

    value 与 expression 二选一；场景相关轴用 expression（B/T/P 等），
    配置常量轴用 value（dim=4096 等）。unknown_reason 仅用于静态确实未知、
    且不是场景可求值的轴（如每专家 token 数）。
    """

    symbol: str
    name: str
    meaning: str
    origin: str
    evidence_ref: str
    value: int | None = None
    expression: str | None = None
    unknown_reason: str | None = None

    def validate(self) -> None:
        try:
            if self.expression is not None:
                check_expression(self.expression)
        except ExpressionError as exc:
            raise RulesError(f"轴 {self.symbol} 的表达式非法: {exc}") from exc
        if self.value is None and self.expression is None and not (
            self.unknown_reason and self.unknown_reason.strip()
        ):
            raise RulesError(f"轴 {self.symbol} 既无值/表达式也无 unknown_reason")


# 场景相关轴：值随场景变化，模板中保存表达式。
_AXIS_B = AxisDef("B", "batch_size", "批大小", "场景输入", "scenarios", expression="B")
_AXIS_T = AxisDef("T", "seq_len", "本次调用序列长度", "场景输入", "scenarios", expression="T")
_AXIS_P = AxisDef("P", "past_len", "调用前已处理长度（历史 cache）", "场景输入", "scenarios", expression="P")
_AXIS_N_TOKENS = AxisDef(
    "N_tokens", "num_tokens", "展平后的 token 数（B×T）", "MoE 前视图 model.py:636",
    "model.py:636", expression="B * T",
)

# 配置常量轴：值来自 run config / ModelArgs，来源逐条记录。
_CONFIG_AXES: tuple[AxisDef, ...] = (
    AxisDef("H", "hidden_size", "隐藏维度", "run config dim", "run_config.json:3", value=4096),
    AxisDef("N_hc", "hc_streams", "mHC 超连接流数", "run config hc_mult", "run_config.json:33", value=4),
    AxisDef("Hq", "query_heads", "注意力查询头数", "run config n_heads", "run_config.json:12", value=64),
    AxisDef("Hkv", "kv_heads", "KV 头数（共享 KV 单投影）", "root config num_key_value_heads", "root_config.json:30", value=1),
    AxisDef("D", "head_dim", "注意力头维度", "run config head_dim", "run_config.json:20", value=512),
    AxisDef("D_rope", "rope_head_dim", "RoPE 旋转维度", "run config rope_head_dim", "run_config.json:21", value=64),
    AxisDef("D_nope", "nope_head_dim", "非 RoPE 头维度（D−D_rope）", "推导 head_dim−rope_head_dim", "model.py:455", value=448),
    AxisDef("Rq", "q_lora_rank", "Q 低秩秩", "run config q_lora_rank", "run_config.json:19", value=1024),
    AxisDef("Ro", "o_lora_rank", "分组低秩 O 投影秩", "run config o_lora_rank", "run_config.json:23", value=1024),
    AxisDef("G", "output_groups", "O 投影分组数", "run config o_groups", "run_config.json:22", value=8),
    AxisDef("W", "window_size", "滑动窗口长度", "run config window_size", "run_config.json:24", value=128),
    AxisDef("Hi", "index_heads", "Indexer 头数", "run config index_n_heads", "run_config.json:30", value=64),
    AxisDef("Di", "index_head_dim", "Indexer 头维度", "run config index_head_dim", "run_config.json:31", value=128),
    AxisDef("Ki", "index_topk", "Indexer 每查询选择上限", "run config index_topk", "run_config.json:32", value=512),
    AxisDef("E", "routed_experts", "路由专家总数", "run config n_routed_experts", "run_config.json:13", value=256),
    AxisDef("K", "experts_per_token", "每 token 激活专家数", "run config n_activated_experts", "run_config.json:15", value=6),
    AxisDef("I", "expert_intermediate", "专家 FFN 中间维度", "run config moe_inter_dim", "run_config.json:4", value=2048),
    AxisDef("V", "vocab_size", "词表大小", "run config vocab_size", "run_config.json:2", value=129280),
    AxisDef("N_tgt", "dspark_target_layers", "DSpark 主干读出目标层数", "run config dspark_target_layer_ids", "run_config.json:10", value=3),
    AxisDef("N_draft", "dspark_block_size", "DSpark 每步草稿 token 数", "run config dspark_block_size", "run_config.json:8", value=5),
    AxisDef("Mcap", "max_seq_len", "KV 缓存序列容量（ModelArgs 默认；run config 未覆盖，interactive 入口覆盖为 65536）", "ModelArgs 默认值", "model.py:38", value=4096),
    # 派生轴（表达式引用其他轴）
    AxisDef("N_mix", "hc_mix_dim", "mHC 混合参数维度 (2+N_hc)×N_hc", "推导 model.py:670", "model.py:670", expression="(2 + N_hc) * N_hc"),
    AxisDef("N_hcH", "hc_stream_hidden", "mHC 展平维度 N_hc×H", "推导 model.py:683", "model.py:683", expression="N_hc * H"),
    AxisDef("D_g", "group_head_dim", "O 分组前每组的紧凑头维度 Hq×D/G", "推导 model.py:542", "model.py:542-546", expression="Hq * D // G"),
    AxisDef("GRo", "group_lora_flat", "O 分组低秩展平维度 G×Ro", "推导 model.py:547", "model.py:546-547", expression="G * Ro"),
    AxisDef("D2", "compress_proj_dim", "压缩投影宽度（overlap 时 2×D）", "推导 model.py:303（coff=1+overlap）", "model.py:298-303", expression="2 * D"),
)

# CSA·TopK 代表层的压缩比（compress_ratios[4]=4）。
_AXIS_R_CSA = AxisDef("r", "compress_ratio", "CSA 压缩比（compress_ratios[layer]=4）", "run config compress_ratios[4]", "run_config.json:39", value=4)

# 静态未知轴：数据相关或历史相关，模板中诚实标记。
_UNKNOWN_AXES: tuple[AxisDef, ...] = (
    AxisDef("N_e", "tokens_per_expert", "路由到单个专家的 token 数", "数据相关（gate 路由结果）", "model.py:639-645", unknown_reason="依赖每 token 的路由选择，静态分析未知"),
    AxisDef("C", "compressed_entries", "压缩 KV 缓存中已完成的条目数", "历史相关（调用序列决定）", "model.py:379-382", unknown_reason="取决于本调用前已完成压缩的条目数；场景求值给出可见上限 (P+T)//r"),
)

AXIS_DEFS: tuple[AxisDef, ...] = (
    _AXIS_B, _AXIS_T, _AXIS_P, _AXIS_N_TOKENS, *_CONFIG_AXES, _AXIS_R_CSA, *_UNKNOWN_AXES,
)

AXIS_BY_SYMBOL: dict[str, AxisDef] = {a.symbol: a for a in AXIS_DEFS}


# ---------------------------------------------------------------------------
# 模型数值（从 run config 读取并交叉核对）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelValues:
    """从 run config 读出、构建图所需的最小数值集合。"""

    dim: int
    n_layers: int
    n_hash_layers: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    window_size: int
    compress_ratios: tuple[int, ...]
    index_topk: int
    index_n_heads: int
    index_head_dim: int
    n_routed_experts: int
    n_activated_experts: int
    moe_inter_dim: int
    n_shared_experts: int
    hc_mult: int
    vocab_size: int
    n_mtp_layers: int
    dspark_block_size: int
    dspark_target_layer_ids: tuple[int, ...]
    dspark_markov_rank: int
    max_seq_len: int  # ModelArgs 默认（AST 读取）；run config 未覆盖
    max_batch_size: int


def _dataclass_defaults(model_py: Path, dataclass_name: str) -> dict[str, int]:
    """AST 读取 ModelArgs dataclass 字段的字面量默认值（int/float/None）。"""
    tree = ast.parse(model_py.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == dataclass_name:
            defaults: dict[str, int] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.value is not None and isinstance(stmt.value, ast.Constant):
                        value = stmt.value.value
                        if value is None or isinstance(value, (int, float)):
                            defaults[stmt.target.id] = value
            return defaults
    raise RulesError(f"{model_py} 中未找到 dataclass {dataclass_name}")


def load_model_values(workspace_root: Path) -> ModelValues:
    """读取 run config 与 ModelArgs 默认值；缺字段/类型不符具名报错。"""
    run_cfg_path = workspace_root / SOURCE_FILES["run_config.json"]
    model_py = workspace_root / SOURCE_FILES["model.py"]
    cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
    defaults = _dataclass_defaults(model_py, "ModelArgs")

    required_int = (
        "dim", "n_layers", "n_hash_layers", "n_heads", "head_dim", "rope_head_dim",
        "q_lora_rank", "o_lora_rank", "o_groups", "window_size", "index_topk",
        "index_n_heads", "index_head_dim", "n_routed_experts", "n_activated_experts",
        "moe_inter_dim", "n_shared_experts", "hc_mult", "vocab_size", "n_mtp_layers",
        "dspark_block_size", "dspark_markov_rank",
    )
    for name in required_int:
        if not isinstance(cfg.get(name), int) or isinstance(cfg.get(name), bool):
            raise RulesError(f"run config 缺少或类型不符的整数字段: {name}")
    ratios = cfg.get("compress_ratios")
    if not isinstance(ratios, list) or not all(isinstance(v, int) for v in ratios):
        raise RulesError("run config compress_ratios 必须是整数数组")
    targets = cfg.get("dspark_target_layer_ids")
    if not isinstance(targets, list) or not all(isinstance(v, int) for v in targets):
        raise RulesError("run config dspark_target_layer_ids 必须是整数数组")

    # 轴字典中的配置常量与实际 config 数值交叉核对；不一致说明轴字典过期。
    checks = {
        "H": cfg["dim"], "N_hc": cfg["hc_mult"], "Hq": cfg["n_heads"], "D": cfg["head_dim"],
        "D_rope": cfg["rope_head_dim"], "D_nope": cfg["head_dim"] - cfg["rope_head_dim"],
        "Rq": cfg["q_lora_rank"], "Ro": cfg["o_lora_rank"], "G": cfg["o_groups"],
        "W": cfg["window_size"], "Hi": cfg["index_n_heads"], "Di": cfg["index_head_dim"],
        "Ki": cfg["index_topk"], "E": cfg["n_routed_experts"], "K": cfg["n_activated_experts"],
        "I": cfg["moe_inter_dim"], "V": cfg["vocab_size"], "N_tgt": len(targets),
        "N_draft": cfg["dspark_block_size"],
    }
    for symbol, actual in checks.items():
        declared = AXIS_BY_SYMBOL[symbol].value
        if declared != actual:
            raise RulesError(
                f"轴字典 {symbol}={declared} 与 run config 实际值 {actual} 不一致；"
                "轴字典需更新"
            )
    for symbol, default_key in (("Mcap", "max_seq_len"), ):
        declared = AXIS_BY_SYMBOL[symbol].value
        if defaults.get(default_key) != declared:
            raise RulesError(
                f"轴字典 {symbol}={declared} 与 ModelArgs 默认值 "
                f"{defaults.get(default_key)!r} 不一致"
            )

    return ModelValues(
        dim=cfg["dim"], n_layers=cfg["n_layers"], n_hash_layers=cfg["n_hash_layers"],
        n_heads=cfg["n_heads"], head_dim=cfg["head_dim"], rope_head_dim=cfg["rope_head_dim"],
        q_lora_rank=cfg["q_lora_rank"], o_lora_rank=cfg["o_lora_rank"],
        o_groups=cfg["o_groups"], window_size=cfg["window_size"],
        compress_ratios=tuple(ratios), index_topk=cfg["index_topk"],
        index_n_heads=cfg["index_n_heads"], index_head_dim=cfg["index_head_dim"],
        n_routed_experts=cfg["n_routed_experts"], n_activated_experts=cfg["n_activated_experts"],
        moe_inter_dim=cfg["moe_inter_dim"], n_shared_experts=cfg["n_shared_experts"],
        hc_mult=cfg["hc_mult"], vocab_size=cfg["vocab_size"],
        n_mtp_layers=cfg["n_mtp_layers"], dspark_block_size=cfg["dspark_block_size"],
        dspark_target_layer_ids=tuple(targets), dspark_markov_rank=cfg["dspark_markov_rank"],
        max_seq_len=int(defaults["max_seq_len"]), max_batch_size=int(defaults["max_batch_size"]),
    )


# ---------------------------------------------------------------------------
# 层分组
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerGroup:
    """一个主干分组：注意力类型 × 门控类型，含成员层与模板状态。"""

    group_id: str
    attention_kind: str  # swa | csa | hca
    gate_kind: str       # hash | topk
    compress_ratio: int
    has_indexer: bool
    member_layers: tuple[int, ...]
    template_status: str  # extracted | pending_extraction（真实提取后才可 extracted）
    note: str


def derive_layer_groups(
    values: ModelValues,
    extracted_groups: frozenset[str] | set[str] = frozenset(),
) -> tuple[LayerGroup, ...]:
    """由 compress_ratios + n_hash_layers 推导四类主干分组。

    规则（evidence model.py:459,472-477,551-561）：
    - compress_ratios[layer]==0 → SWA（纯滑动窗口，无压缩/Indexer）
    - ==4 → CSA（overlap 压缩 + 学习式 Indexer）
    - ==128 → HCA（压缩但无学习式 Indexer）
    - layer < n_hash_layers → Hash 门控（token ID 查表），否则 Top-K 门控
    主干范围只覆盖前 n_layers 项；DSpark stage（model.py:750-753
    DSparkAttention.forward 首行 `assert self.compress_ratio == 0`，
    stage 层号 = n_layers + stage_id，压缩比取 compress_ratios 尾部）
    不参与主干分组，其尾部 ratio 异常单独具名报错。
    extracted_groups 只能由真实提取器在对应模板图构建并核验成功后传入；
    默认空集——任何分组都不会仅因声明而标 extracted。
    推导后校验：并集 == 0..n_layers-1、互不重叠、每组内 compress_ratio 一致。
    """
    ratios = values.compress_ratios
    if len(ratios) < values.n_layers:
        raise RulesError(
            f"compress_ratios 长度 {len(ratios)} 少于 n_layers={values.n_layers}"
        )
    grouped: dict[tuple[str, str], list[int]] = {}
    layer_ratio: dict[int, int] = {}
    for layer in range(values.n_layers):
        ratio = ratios[layer]
        if ratio == 0:
            att = "swa"
        elif ratio == 4:
            att = "csa"
        elif ratio == 128:
            att = "hca"
        else:
            raise RulesError(
                f"层 {layer} 的 compress_ratio={ratio} 不属于已知集合 {{0,4,128}}；"
                "不能归类，请先人工核对源码"
            )
        gate = "hash" if layer < values.n_hash_layers else "topk"
        grouped.setdefault((att, gate), []).append(layer)
        layer_ratio[layer] = ratio

    allowed_combinations = {
        ("swa", "hash"), ("csa", "hash"), ("hca", "topk"), ("csa", "topk"),
    }
    unsupported = set(grouped) - allowed_combinations
    if unsupported:
        details = ", ".join(
            f"{att}_{gate}（层 {grouped[(att, gate)]}）"
            for att, gate in sorted(unsupported)
        )
        raise RulesError(
            f"出现未支持的主干组合（本切片只支持 "
            f"swa_hash/csa_hash/hca_topk/csa_topk）: {details}"
        )

    # 尾部 DSpark stage 的 compress_ratio 检查（不进分组，只验值合理）：
    # 源码 DSparkAttention.forward 要求 ratio==0（model.py:753）。
    tail = ratios[values.n_layers:]
    for offset, ratio in enumerate(tail):
        if ratio != 0:
            raise RulesError(
                f"DSpark stage {offset}（层 {values.n_layers + offset}）的 "
                f"compress_ratio={ratio} 不为 0；源码 DSparkAttention 要求 "
                "compress_ratio == 0（model.py:753）"
            )

    notes = {
        ("swa", "hash"): "纯滑动窗口 + Hash 门控；SWA 关闭 YaRN、使用基础 rope_theta（model.py:484-485）",
        ("csa", "hash"): "overlap 压缩 + 学习式 Indexer + Hash 门控",
        ("hca", "topk"): "压缩（r=128，无 overlap）+ 无学习式 Indexer，使用固定压缩位置选择（model.py:519）",
        ("csa", "topk"): "overlap 压缩 + 学习式 Indexer + Top-K 门控；本切片的完整提取代表",
    }
    groups = []
    for (att, gate), members in sorted(grouped.items()):
        ratios_in_group = {layer_ratio[m] for m in members}
        if len(ratios_in_group) != 1:
            raise RulesError(f"分组 {att}_{gate} 内 compress_ratio 不一致: {sorted(ratios_in_group)}")
        group_id = f"{att}_{gate}"
        groups.append(LayerGroup(
            group_id=group_id, attention_kind=att, gate_kind=gate,
            compress_ratio=ratios_in_group.pop(), has_indexer=(att == "csa"),
            member_layers=tuple(members),
            template_status=(
                "extracted" if group_id in extracted_groups else "pending_extraction"
            ),
            note=notes[(att, gate)],
        ))

    covered = sorted(m for g in groups for m in g.member_layers)
    if covered != list(range(values.n_layers)):
        raise RulesError(
            f"层分组并集 {covered} ≠ 0..{values.n_layers - 1}；分组推导有遗漏或重叠"
        )
    derived_ids = {f"{att}_{gate}" for att, gate in grouped}
    unknown_extracted = set(extracted_groups) - derived_ids
    if unknown_extracted:
        raise RulesError(
            f"extracted_groups 声明了推导中不存在的分组: "
            f"{sorted(unknown_extracted)}；extracted 只能标记真实推导出的分组"
        )
    return tuple(groups)


# ---------------------------------------------------------------------------
# 守卫目录
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardDef:
    """一条静态条件：白名单谓词表达式 + 源码依据。"""

    stable_id: str
    description: str
    predicate_expr: str
    evidence_ref: str

    def validate(self) -> None:
        try:
            check_expression(self.predicate_expr)
        except ExpressionError as exc:
            raise RulesError(f"守卫 {self.stable_id} 的谓词非法: {exc}") from exc


GUARD_DEFS: tuple[GuardDef, ...] = (
    GuardDef("g_prefill", "prefill 阶段（整段序列一次前向）", "start_pos == 0", "model.py:331,523"),
    GuardDef("g_decode", "decode 阶段（单 token 增量前向）", "start_pos > 0", "model.py:349,534"),
    GuardDef("g_compress_csa", "本层启用 KV 压缩（CSA/HCA）", "r > 0", "model.py:472"),
    GuardDef("g_has_indexer", "学习式 Indexer 存在（CSA 专属，r==4）", "r == 4", "model.py:474-475"),
    GuardDef("g_compress_prefill", "prefill 压缩触发：seqlen ≥ ratio", "start_pos == 0 and T >= r", "model.py:332"),
    GuardDef("g_compress_prefill_skip", "prefill 压缩未触发（T 不足一个压缩块，compressor 返回 None）", "start_pos == 0 and T < r", "model.py:332,366-367"),
    GuardDef("g_compress_prefill_remainder", "prefill 尾块存在：remainder = T % r > 0（尾块写状态槽，与是否触发压缩无关）", "start_pos == 0 and T % r > 0", "model.py:339-341"),
    GuardDef("g_compress_decode", "decode 压缩触发：(start_pos+1) 能整除 ratio", "(start_pos + 1) % r == 0", "model.py:350"),
    GuardDef("g_decode_no_compress", "decode 压缩未触发：decode 相位且 (start_pos+1) 不整除 ratio（状态只累积，不池化）", "start_pos > 0 and (start_pos + 1) % r != 0", "model.py:350,366-367"),
    GuardDef("g_window_wrap", "prefill 超窗：序列长于窗口时环形写入", "T > W", "model.py:524-528"),
    GuardDef("g_causal_mask", "prefill 压缩条目因果掩码", "start_pos == 0", "model.py:430-432"),
)


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioDef:
    """一个分析场景：B/T/P/start_pos 环境 + 覆盖目的说明。"""

    scenario_id: str
    description: str
    env: dict[str, int] = field(default_factory=dict)
    covers: tuple[str, ...] = ()


def _scenario_env(batch: int, t_len: int, past: int) -> dict[str, int]:
    return {"B": batch, "T": t_len, "P": past, "start_pos": past}


SCENARIOS: tuple[ScenarioDef, ...] = (
    ScenarioDef("prefill_t3", "prefill：3 token（不足一个压缩块，T<r）", _scenario_env(1, 3, 0), ("压缩不触发", "纯窗口写入")),
    ScenarioDef("prefill_t4", "prefill：4 token（恰好一个压缩块）", _scenario_env(1, 4, 0), ("压缩边界：恰好 1 条",)),
    ScenarioDef("prefill_t5", "prefill：5 token（1 条压缩 + 1 条 remainder 暂存）", _scenario_env(1, 5, 0), ("压缩边界：remainder>0",)),
    ScenarioDef("prefill_t127", "prefill：127 token（31 条压缩 + 3 条 remainder）", _scenario_env(1, 127, 0), ("压缩边界", "未凑满块")),
    ScenarioDef("prefill_t128", "prefill：128 token（32 条压缩、恰好覆盖窗口长度）", _scenario_env(1, 128, 0), ("窗口边界 T==W",)),
    ScenarioDef("prefill_t129", "prefill：129 token（32 条压缩 + 1 remainder；T>W 环形写入）", _scenario_env(1, 129, 0), ("窗口回绕",)),
    ScenarioDef("decode_p2", "decode：P=2（早期，未触发压缩；end_pos=3，Indexer 可见 0 条压缩）", _scenario_env(1, 1, 2), ("decode 压缩未触发", "Indexer C=0 边界")),
    ScenarioDef("decode_p3", "decode：P=3（本步触发压缩写入第 1 条；end_pos=4，Indexer 可见 1 条）", _scenario_env(1, 1, 3), ("decode 压缩触发", "Indexer 首条压缩条目可见")),
    ScenarioDef("decode_p4", "decode：P=4（未触发压缩；end_pos=5，Indexer 可见 1 条）", _scenario_env(1, 1, 4), ("decode 压缩未触发", "Indexer 少于 Ki")),
    ScenarioDef("decode_p126", "decode：P=126（HCA 边界：127 不整除 128，不触发；CSA (127)%4=3 也不触发）", _scenario_env(1, 1, 126), ("HCA 未触发边界", "decode 压缩未触发")),
    ScenarioDef("decode_p127", "decode：P=127（本步触发 CSA 压缩：128 能整除 4；HCA 128 能整除 128 也触发）", _scenario_env(1, 1, 127), ("decode 压缩触发",)),
    ScenarioDef("decode_p128", "decode：P=128（窗口回绕槽位 0；不触发压缩）", _scenario_env(1, 1, 128), ("窗口回绕", "decode 压缩未触发")),
)


def scenario_base_env(values: ModelValues, scenario: ScenarioDef) -> dict[str, int]:
    """场景求值环境：场景变量 + 模型常量（与轴字典同源）。"""
    env = dict(scenario.env)
    env.update({
        "H": values.dim, "N_hc": values.hc_mult, "Hq": values.n_heads, "Hkv": 1,
        "D": values.head_dim, "D_rope": values.rope_head_dim,
        "D_nope": values.head_dim - values.rope_head_dim,
        "Rq": values.q_lora_rank, "Ro": values.o_lora_rank, "G": values.o_groups,
        "W": values.window_size, "Hi": values.index_n_heads, "Di": values.index_head_dim,
        "Ki": values.index_topk, "E": values.n_routed_experts,
        "K": values.n_activated_experts, "I": values.moe_inter_dim,
        "V": values.vocab_size, "N_tgt": len(values.dspark_target_layer_ids),
        "N_draft": values.dspark_block_size, "Mcap": values.max_seq_len,
    })
    return env
