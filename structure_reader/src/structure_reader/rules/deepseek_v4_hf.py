"""DeepSeek-V4-Flash HF 有效配置解释（新 HF 主干，旧 rules/deepseek_v4_flash_0731.py 保持历史身份）。

只做静态解释：读根 config.json，按本地官方 Transformers 检出中
``configuration_deepseek_v4.py::DeepseekV4Config.__post_init__`` 与
``configuration_utils.py/modeling_rope_utils.py`` 的标准化规则，推导
HF 实例化配置类时实际生效的参数。不导入 torch/transformers、不实例化
配置类、不运行模型构造程序——每条派生规则以源码行区间为据。

产出：
- :class:`HfModelValues`：layer_idx 视角的结构参数（含属性访问与派生）。
- :func:`load_hf_model_values`：从工作区读取并解释。
- :func:`effective_parameter_records`：逐参数 raw 字段/默认/别名/派生规则与来源
  （供 reading_document 的参数符号区与来源清单）。
- ``SOURCE_FILES``：所选 forward 路径实际调用的源码/配置闭包（相对工作区根）。

legacy 转换事实（configuration_deepseek_v4.py，2026-10-17 检出）：
- ``compress_ratios``(46 项) → ``_COMPRESS_RATIO_TO_LAYER_TYPE``{0,4,128} → layer_types，
  再截取前 num_hidden_layers(43) 项（:254-262）。
- ``num_hash_layers`` → 前 n 项 hash_moe + 其余 moe → mlp_layer_types（:264-268）。
- ``qk_rope_head_dim`` → partial_rotary_factor = qk_rope_head_dim/head_dim（:270-277），
  运行时属性 qk_rope_head_dim = int(head_dim*partial_rotary_factor)。
- ``attribute_map``：num_local_experts→n_routed_experts、intermediate_size→moe_intermediate_size
  （configuration_deepseek_v4.py:104-107；消费点为 modeling_deepseek_v4.py:1030/1032/1011）。
- rope_parameters：convert_rope_params_to_dict 把 flat rope_scaling 标准化（type→rope_type、
  补 rope_theta/original_max_position_embeddings），随后 __post_init__ :288-307 拆成
  main(default, theta=rope_theta) / compress(**yarn, theta=compress_rope_theta,
  attention_factor 强制 1.0)。
- 不存在的源码调用依据的 dspark_* / num_nextn_predict_layers 字段不转成计算结构。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "HfConfigError",
    "HfModelValues",
    "SOURCE_FILES",
    "LAYER_TYPE_BY_RATIO",
    "ANALYZED_SOURCE_FINGERPRINTS",
    "verify_source_fingerprints",
    "load_hf_model_values",
    "effective_parameter_records",
    "effective_layer_types",
]

# 所选 eager forward 路径实际读取的定义闭包（相对工作区根；相对 structure_reader 根的
# 键名供 evidence/来源记录使用）。modeling/configuration 之外只纳入影响当前结构/
# 维度/缓存的四个定义（cache_utils、masking_utils、modeling_rope_utils、activations）。
SOURCE_FILES = {
    "config.json": "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json",
    "modeling_deepseek_v4.py": (
        "hflib/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py"
    ),
    "configuration_deepseek_v4.py": (
        "hflib/transformers/src/transformers/models/deepseek_v4/configuration_deepseek_v4.py"
    ),
    "cache_utils.py": "hflib/transformers/src/transformers/cache_utils.py",
    "masking_utils.py": "hflib/transformers/src/transformers/masking_utils.py",
    "modeling_rope_utils.py": "hflib/transformers/src/transformers/modeling_rope_utils.py",
    "activations.py": "hflib/transformers/src/transformers/activations.py",
}

# configuration_deepseek_v4.py:28-32
LAYER_TYPE_BY_RATIO = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}

# 已分析源码指纹（SHA256，2026-10-17 检出 536ecc00… 上的实际值，经复算核对）。
# 这些指纹不是"永远正确"的期望：来源漂移（任一文件变化）时构建具名拒绝、
# 要求重新分析后更新指纹，而不是拿旧结论继续产出。config.json 的指纹同样
# 固定——配置变化即样板语义可能变化，同样拒绝。
ANALYZED_SOURCE_FINGERPRINTS = {
    "config.json": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
    "modeling_deepseek_v4.py": "d6eb834f9d10eca879ca5390f219259eb452d19e1cf72d827f4334cbe4682266",
    "configuration_deepseek_v4.py": "49c818404caff1b4e76dff0ae7f468512932a5c40cc41b6cbd6585da4dec3aa2",
    "cache_utils.py": "2bef552bc39e8a4dc7ddfcf25740e6a4a7c4b836d92f5e56629a8ebfcd9cb246",
    "masking_utils.py": "c1d0e1e61754db819382bb03acdd2ab38c1bed3e9e9cf010918d2bf2771a8415",
    "modeling_rope_utils.py": "6a0727073e080e163d0287e15fc1acb90035e9a9df9cb19a13b632a40abfac24",
    "activations.py": "5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665",
}


def verify_source_fingerprints(workspace_root: Path) -> None:
    """复算 SOURCE_FILES 各输入 SHA256 并与已分析指纹核对；漂移具名拒绝。

    已分析指纹不自动接受新值：来源变化意味着行区间/语义结论可能失效，
    必须显式重新分析并更新 ANALYZED_SOURCE_FINGERPRINTS 后才可产出。
    """
    import hashlib

    for key, relpath in SOURCE_FILES.items():
        path = workspace_root / relpath
        if not path.is_file():
            raise HfConfigError(f"已分析输入缺失: {relpath}（不下载、不伪造）")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = ANALYZED_SOURCE_FINGERPRINTS.get(key)
        if expected is None:
            raise HfConfigError(f"输入 {key} 未登记已分析指纹（拒绝未分析来源）")
        if actual != expected:
            raise HfConfigError(
                f"来源漂移: {relpath} 实际 SHA256 {actual} ≠ 已分析指纹 {expected}；"
                "来源已变化，既有分析结论作废，需重新分析并更新指纹后才可构建"
            )


class HfConfigError(ValueError):
    """HF 根配置缺失/非法，或与已解释的身份不一致（具名拒绝）。"""


def _load_root_config(workspace_root: Path) -> dict:
    path = workspace_root / SOURCE_FILES["config.json"]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HfConfigError(f"HF 根配置不可读: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HfConfigError(f"HF 根配置不是合法 JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("model_type") != "deepseek_v4":
        raise HfConfigError(f"HF 根配置 model_type 必须是 deepseek_v4: {path}")
    if raw.get("architectures") != ["DeepseekV4ForCausalLM"]:
        raise HfConfigError(
            f"HF 根配置 architectures 必须是 ['DeepseekV4ForCausalLM']: {path}"
        )
    return raw


def effective_layer_types(raw: dict) -> list[str]:
    """legacy compress_ratios → layer_types 并按 HF 截取前 num_hidden_layers 项。

    configuration_deepseek_v4.py:254-262：显式 layer_types > legacy compress_ratios >
    V4-Pro 默认；本配置只提供 legacy 数组。未知 ratio 值具名拒绝，不猜层型。
    """
    ratios = raw.get("compress_ratios")
    if not isinstance(ratios, list) or not ratios:
        raise HfConfigError("根配置缺少 compress_ratios（legacy 层型来源）")
    n = raw["num_hidden_layers"]
    if len(ratios) < n:
        raise HfConfigError(
            f"compress_ratios 长度 {len(ratios)} 少于 num_hidden_layers={n}"
        )
    types = []
    for index, ratio in enumerate(ratios):
        layer_type = LAYER_TYPE_BY_RATIO.get(ratio)
        if layer_type is None:
            raise HfConfigError(
                f"compress_ratios[{index}]={ratio!r} 不在 "
                f"{sorted(LAYER_TYPE_BY_RATIO)}（_COMPRESS_RATIO_TO_LAYER_TYPE）内"
            )
        types.append(layer_type)
    return types[:n]


def effective_mlp_layer_types(raw: dict) -> list[str]:
    """num_hash_layers → 前 n_hash 项 hash_moe + 其余 moe。

    configuration_deepseek_v4.py:264-268（mlp_layer_types 未显式提供时）。
    """
    n = raw["num_hidden_layers"]
    n_hash = raw.get("num_hash_layers")
    if n_hash is None:
        n_hash = 3  # default_num_hash_layers（configuration_deepseek_v4.py:151）
    if not isinstance(n_hash, int) or n_hash < 0:
        raise HfConfigError(f"num_hash_layers 非法: {n_hash!r}")
    n_hash = min(n_hash, n)
    return ["hash_moe"] * n_hash + ["moe"] * (n - n_hash)


@dataclass(frozen=True)
class HfModelValues:
    """layer_idx 视角的 HF 有效配置值（全部来自静态解释，非运行时探值）。"""

    # 身份
    model_key: str
    display_name: str
    implementation: str
    source_revision: str
    config_path: str
    # 全局
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    hidden_act: str
    use_cache: bool
    sliding_window: int
    max_position_embeddings: int
    # 注意力低秩/分组
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    # RoPE
    qk_rope_head_dim: int
    partial_rotary_factor: float
    rope_main_theta: float
    rope_compress_theta: float
    rope_compress_factor: int
    rope_compress_beta_fast: int
    rope_compress_beta_slow: int
    rope_compress_original_max_position_embeddings: int
    rope_compress_attention_factor: float
    # CSA/HCA 压缩与 Indexer
    compress_rate_csa: int
    compress_rate_hca: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    # mHC
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    # MoE
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    scoring_func: str
    norm_topk_prob: bool
    routed_scaling_factor: float
    moe_intermediate_size: int
    swiglu_limit: float
    # 层规划
    layer_types: list[str]
    mlp_layer_types: list[str]
    # 声明但明确不在本样板范围
    quantization_config: dict
    num_nextn_predict_layers: int

    def layer_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def mlp_layer_type(self, layer_idx: int) -> str:
        return self.mlp_layer_types[layer_idx]


def load_hf_model_values(
    workspace_root: str | Path,
    model_key: str = "deepseek_v4_flash_0731_hf",
) -> HfModelValues:
    """读取根 config.json 并静态解释为 HF 有效配置；失配/缺字段具名拒绝。"""
    workspace_root = Path(workspace_root)
    raw = _load_root_config(workspace_root)

    def _require(name: str):
        value = raw.get(name)
        if value is None:
            raise HfConfigError(f"根配置缺少必要字段 {name!r}")
        return value

    layer_types = effective_layer_types(raw)
    mlp_types = effective_mlp_layer_types(raw)

    # partial_rotary_factor：__post_init__ :270-278。legacy qk_rope_head_dim 优先。
    head_dim = _require("head_dim")
    qk_rope_raw = raw.get("qk_rope_head_dim")
    partial = (
        qk_rope_raw / head_dim
        if qk_rope_raw is not None
        else 64 / 512  # default_partial_rotary_factor（configuration:133）
    )
    qk_rope_head_dim = int(head_dim * partial)

    # rope_parameters：configuration_utils/modeling_rope_utils 标准化 + __post_init__ :288-307。
    rope_scaling = raw.get("rope_scaling") or {}
    rope_compress_attention_factor = 1.0  # yarn 时 setdefault（configuration:304-306）
    rope_theta = _require("rope_theta")
    compress_theta = _require("compress_rope_theta")

    checkout_root = workspace_root / "hflib" / "transformers"
    source_revision = _git_head(checkout_root)

    return HfModelValues(
        model_key=model_key,
        display_name="DeepSeek-V4-Flash-0731 (HF transformers)",
        implementation="transformers",
        source_revision=source_revision,
        config_path=SOURCE_FILES["config.json"],
        hidden_size=_require("hidden_size"),
        num_hidden_layers=_require("num_hidden_layers"),
        num_attention_heads=_require("num_attention_heads"),
        num_key_value_heads=_require("num_key_value_heads"),
        head_dim=head_dim,
        vocab_size=_require("vocab_size"),
        rms_norm_eps=_require("rms_norm_eps"),
        hidden_act=raw.get("hidden_act", "silu"),
        use_cache=raw.get("use_cache", True),
        sliding_window=_require("sliding_window"),
        max_position_embeddings=_require("max_position_embeddings"),
        q_lora_rank=_require("q_lora_rank"),
        o_groups=_require("o_groups"),
        o_lora_rank=_require("o_lora_rank"),
        qk_rope_head_dim=qk_rope_head_dim,
        partial_rotary_factor=partial,
        rope_main_theta=rope_theta,
        rope_compress_theta=compress_theta,
        rope_compress_factor=rope_scaling.get("factor", 16),
        rope_compress_beta_fast=rope_scaling.get("beta_fast", 32),
        rope_compress_beta_slow=rope_scaling.get("beta_slow", 1),
        rope_compress_original_max_position_embeddings=rope_scaling.get(
            "original_max_position_embeddings", 65536
        ),
        rope_compress_attention_factor=rope_compress_attention_factor,
        # compress_rates 默认字典（configuration:145）；legacy 标量覆盖未出现在本配置
        compress_rate_csa=4,
        compress_rate_hca=128,
        index_n_heads=_require("index_n_heads"),
        index_head_dim=_require("index_head_dim"),
        index_topk=_require("index_topk"),
        hc_mult=_require("hc_mult"),
        hc_sinkhorn_iters=_require("hc_sinkhorn_iters"),
        hc_eps=_require("hc_eps"),
        # attribute_map（configuration:104-107）：num_local_experts→n_routed_experts
        n_routed_experts=_require("n_routed_experts"),
        n_shared_experts=raw.get("n_shared_experts", 1),
        num_experts_per_tok=_require("num_experts_per_tok"),
        scoring_func=_require("scoring_func"),
        norm_topk_prob=raw.get("norm_topk_prob", True),
        routed_scaling_factor=_require("routed_scaling_factor"),
        moe_intermediate_size=_require("moe_intermediate_size"),
        swiglu_limit=_require("swiglu_limit"),
        layer_types=layer_types,
        mlp_layer_types=mlp_types,
        quantization_config=raw.get("quantization_config", {}),
        num_nextn_predict_layers=raw.get("num_nextn_predict_layers", 1),
    )


def _git_head(checkout_root: Path) -> str:
    """本地 Transformers 检出的 HEAD 短身份；不可得时如实标注 unknown。"""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def effective_parameter_records(values: HfModelValues) -> list[dict]:
    """逐参数 raw 字段/默认/别名/派生规则与来源（供参数符号区联查）。

    返回 [{field, value, rule, source_refs:[{path, locator, kind, note}]}]；
    kind ∈ config/default/derived/source，与 hf_reading_contract_v1 一致。
    """
    modeling = SOURCE_FILES["modeling_deepseek_v4.py"]
    configuration = SOURCE_FILES["configuration_deepseek_v4.py"]
    config_json = SOURCE_FILES["config.json"]
    rope_utils = SOURCE_FILES["modeling_rope_utils.py"]
    activations = SOURCE_FILES["activations.py"]
    cache_utils = SOURCE_FILES["cache_utils.py"]

    def cfg(locator: str, note: str) -> dict:
        return {"path": config_json, "locator": locator, "kind": "config", "note": note}

    def rule(locator: str, note: str) -> dict:
        return {"path": configuration, "locator": locator, "kind": "derived", "note": note}

    def src(path: str, locator: str, note: str) -> dict:
        return {"path": path, "locator": locator, "kind": "source", "note": note}

    records = [
        {
            "field": "hidden_size",
            "value": values.hidden_size,
            "rule": "根配置直接字段；HF 配置类默认 4096（本配置显式提供）",
            "source_refs": [cfg("hidden_size", "根配置字段"), rule("125", "配置类字段声明")],
        },
        {
            "field": "num_hidden_layers",
            "value": values.num_hidden_layers,
            "rule": "根配置直接字段；主干层数，layer_types/mlp_layer_types 截取基准",
            "source_refs": [cfg("num_hidden_layers", "根配置字段"), rule("128", "配置类字段声明")],
        },
        {
            "field": "num_attention_heads",
            "value": values.num_attention_heads,
            "rule": "根配置直接字段",
            "source_refs": [cfg("num_attention_heads", "根配置字段"), rule("129", "配置类字段声明")],
        },
        {
            "field": "num_key_value_heads",
            "value": values.num_key_value_heads,
            "rule": "根配置直接字段；=1 即共享 KV MQA（Attention.num_key_value_groups=num_attention_heads）",
            "source_refs": [
                cfg("num_key_value_heads", "根配置字段"),
                src(modeling, "776", "self.num_key_value_groups = config.num_attention_heads（单 KV 头广播）"),
            ],
        },
        {
            "field": "head_dim",
            "value": values.head_dim,
            "rule": "根配置直接字段",
            "source_refs": [cfg("head_dim", "根配置字段"), rule("131", "配置类字段声明")],
        },
        {
            "field": "vocab_size",
            "value": values.vocab_size,
            "rule": "根配置直接字段；embedding/lm_head 词表规模",
            "source_refs": [cfg("vocab_size", "根配置字段"), rule("125", "配置类字段声明")],
        },
        {
            "field": "rms_norm_eps",
            "value": values.rms_norm_eps,
            "rule": "根配置直接字段；全部 RMSNorm 的 epsilon",
            "source_refs": [cfg("rms_norm_eps", "根配置字段"), rule("167", "配置类字段声明")],
        },
        {
            "field": "hidden_act",
            "value": values.hidden_act,
            "rule": "根配置直接字段；SiLU（专家/共享专家激活）",
            "source_refs": [
                cfg("hidden_act", "根配置字段"),
                src(activations, "ACT2FN['silu']", "SiLUActivation"),
            ],
        },
        {
            "field": "use_cache",
            "value": values.use_cache,
            "rule": "根配置直接字段；本分析启用 KVCache",
            "source_refs": [cfg("use_cache", "根配置字段"), rule("168", "配置类字段声明")],
        },
        {
            "field": "sliding_window",
            "value": values.sliding_window,
            "rule": "根配置直接字段；滑动窗口 n_win（含 CSA/HCA 主 KV 滑动分支）",
            "source_refs": [
                cfg("sliding_window", "根配置字段"),
                src(cache_utils, "211-294", "DynamicSlidingWindowLayer：持久保存 sliding_window-1 尾部"),
            ],
        },
        {
            "field": "max_position_embeddings",
            "value": values.max_position_embeddings,
            "rule": "根配置直接字段；RoPE 位置上限（rope 缓冲 max_seq_len_cached）",
            "source_refs": [cfg("max_position_embeddings", "根配置字段"), rule("140", "配置类字段声明")],
        },
        {
            "field": "q_lora_rank",
            "value": values.q_lora_rank,
            "rule": "根配置直接字段；低秩 Q 中间秩（q_a_proj 输出维度）",
            "source_refs": [
                cfg("q_lora_rank", "根配置字段"),
                src(modeling, "783-786", "q_a_proj/q_a_norm/q_b_proj 消费点"),
            ],
        },
        {
            "field": "o_groups",
            "value": values.o_groups,
            "rule": "根配置直接字段；分组输出投影组数 g",
            "source_refs": [
                cfg("o_groups", "根配置字段"),
                src(modeling, "789-791", "o_a_proj = DeepseekV4GroupedLinear(...) 消费点"),
            ],
        },
        {
            "field": "o_lora_rank",
            "value": values.o_lora_rank,
            "rule": "根配置直接字段；每组中间维度 d_g",
            "source_refs": [cfg("o_lora_rank", "根配置字段"), rule("155", "配置类字段声明")],
        },
        {
            "field": "qk_rope_head_dim",
            "value": values.qk_rope_head_dim,
            "rule": "legacy 字段 → partial_rotary_factor=qk_rope_head_dim/head_dim，"
            "再 qk_rope_head_dim=int(head_dim*partial_rotary_factor)；运行时属性",
            "source_refs": [
                cfg("qk_rope_head_dim", "legacy 字段（runtime-only，非数据类字段）"),
                rule("270-278", "__post_init__ legacy 转换"),
            ],
        },
        {
            "field": "partial_rotary_factor",
            "value": values.partial_rotary_factor,
            "rule": "派生 = qk_rope_head_dim/head_dim = 64/512；进入 main/compress 两组 rope_parameters",
            "source_refs": [rule("270-278", "__post_init__ 派生")],
        },
        {
            "field": "rope_parameters.main",
            "value": {
                "rope_type": "default",
                "rope_theta": values.rope_main_theta,
                "partial_rotary_factor": values.partial_rotary_factor,
            },
            "rule": "main RoPE：default 型、theta=rope_theta、partial_rotary_factor 继承；"
            "sliding_attention 层与 Attention 主干 rope_layer_type='main'",
            "source_refs": [
                cfg("rope_theta", "根配置字段"),
                rule("293-298", "__post_init__ main 构造"),
                src(rope_utils, "112-138", "DeepseekV4RotaryEmbedding.compute_default_rope_parameters"),
            ],
        },
        {
            "field": "rope_parameters.compress",
            "value": {
                "rope_type": "yarn",
                "rope_theta": values.rope_compress_theta,
                "partial_rotary_factor": values.partial_rotary_factor,
                "factor": values.rope_compress_factor,
                "beta_fast": values.rope_compress_beta_fast,
                "beta_slow": values.rope_compress_beta_slow,
                "original_max_position_embeddings": values.rope_compress_original_max_position_embeddings,
                "attention_factor": values.rope_compress_attention_factor,
            },
            "rule": "compress RoPE：**legacy rope_scaling(**yarn 各键) + theta=compress_rope_theta + "
            "attention_factor 强制 1.0（V4 参考不乘 YaRN mscale）；CSA/HCA 层与压缩器共用",
            "source_refs": [
                cfg("compress_rope_theta", "根配置字段"),
                cfg("rope_scaling", "legacy flat dict（type=yarn 等）"),
                rule("288-307", "__post_init__ compress 构造与 attention_factor 强制"),
                src(rope_utils, "345-483", "_compute_yarn_parameters inv_freq"),
            ],
        },
        {
            "field": "compress_rates",
            "value": {"compressed_sparse_attention": values.compress_rate_csa,
                      "heavily_compressed_attention": values.compress_rate_hca},
            "rule": "默认字典 default_compress_rates（本配置无 compress_rate_csa/hca legacy 覆盖）",
            "source_refs": [rule("145/244-250", "配置类默认 + legacy 标量覆盖点")],
        },
        {
            "field": "index_n_heads",
            "value": values.index_n_heads,
            "rule": "根配置直接字段；Lightning Indexer 查询头数",
            "source_refs": [cfg("index_n_heads", "根配置字段"), rule("156", "配置类字段声明")],
        },
        {
            "field": "index_head_dim",
            "value": values.index_head_dim,
            "rule": "根配置直接字段；Indexer 头维",
            "source_refs": [cfg("index_head_dim", "根配置字段"), rule("157", "配置类字段声明")],
        },
        {
            "field": "index_topk",
            "value": values.index_topk,
            "rule": "根配置直接字段；每查询保留的压缩条目 top-k",
            "source_refs": [cfg("index_topk", "根配置字段"), rule("158", "配置类字段声明")],
        },
        {
            "field": "hc_mult",
            "value": values.hc_mult,
            "rule": "根配置直接字段；mHC 流数 N",
            "source_refs": [
                cfg("hc_mult", "根配置字段"),
                src(modeling, "926-938", "HyperConnection 权重形状 (2+N)*N × N*hidden"),
            ],
        },
        {
            "field": "hc_sinkhorn_iters",
            "value": values.hc_sinkhorn_iters,
            "rule": "根配置直接字段；comb 双随机投影迭代次数",
            "source_refs": [cfg("hc_sinkhorn_iters", "根配置字段"), rule("148", "配置类字段声明")],
        },
        {
            "field": "hc_eps",
            "value": values.hc_eps,
            "rule": "根配置直接字段；Sinkhorn 数值下限",
            "source_refs": [cfg("hc_eps", "根配置字段"), rule("149", "配置类字段声明")],
        },
        {
            "field": "n_routed_experts",
            "value": values.n_routed_experts,
            "rule": "根配置字段 n_routed_experts（attribute_map 把标准名 num_local_experts 映射到它）",
            "source_refs": [
                cfg("n_routed_experts", "根配置字段"),
                rule("104-107", "attribute_map"),
                src(modeling, "1030", "self.num_experts = config.num_local_experts（经 attribute_map 读到本字段）"),
            ],
        },
        {
            "field": "n_shared_experts",
            "value": values.n_shared_experts,
            "rule": "根配置直接字段；共享专家个数（=1 个 DeepseekV4MLP 实例）",
            "source_refs": [
                cfg("n_shared_experts", "根配置字段"),
                src(modeling, "1123", "self.shared_experts = DeepseekV4MLP(config)"),
            ],
        },
        {
            "field": "num_experts_per_tok",
            "value": values.num_experts_per_tok,
            "rule": "根配置直接字段；每 token TopK 选择数（与专家总数不同参数）",
            "source_refs": [cfg("num_experts_per_tok", "根配置字段"), rule("134", "配置类字段声明")],
        },
        {
            "field": "scoring_func",
            "value": values.scoring_func,
            "rule": "根配置直接字段；路由打分激活 sqrt(softplus(x))",
            "source_refs": [
                cfg("scoring_func", "根配置字段"),
                src(activations, "217-221", "SqrtSoftplusActivation"),
            ],
        },
        {
            "field": "norm_topk_prob",
            "value": values.norm_topk_prob,
            "rule": "根配置直接字段；TopK 路由权重归一开关（forward 中固定执行归一）",
            "source_refs": [cfg("norm_topk_prob", "根配置字段"), rule("138", "配置类字段声明")],
        },
        {
            "field": "routed_scaling_factor",
            "value": values.routed_scaling_factor,
            "rule": "根配置直接字段；路由权重缩放",
            "source_refs": [cfg("routed_scaling_factor", "根配置字段"), rule("139", "配置类字段声明")],
        },
        {
            "field": "moe_intermediate_size",
            "value": values.moe_intermediate_size,
            "rule": "根配置字段 moe_intermediate_size（attribute_map 把标准名 intermediate_size 映射到它；共享专家宽度消费点）",
            "source_refs": [
                cfg("moe_intermediate_size", "根配置字段"),
                rule("104-107", "attribute_map"),
                src(modeling, "1011/1032", "config.intermediate_size → moe_intermediate_size"),
            ],
        },
        {
            "field": "swiglu_limit",
            "value": values.swiglu_limit,
            "rule": "根配置直接字段；专家 gate/up 预激活 clamp 上限",
            "source_refs": [cfg("swiglu_limit", "根配置字段"), rule("152", "配置类字段声明")],
        },
        {
            "field": "layer_types",
            "value": values.layer_types,
            "rule": "legacy compress_ratios(46 项) 映射后截取前 num_hidden_layers(43) 项",
            "source_refs": [
                cfg("compress_ratios", "legacy 46 项数组"),
                rule("254-262", "__post_init__ layer_types 派生与截取"),
                rule("28-32", "_COMPRESS_RATIO_TO_LAYER_TYPE"),
            ],
        },
        {
            "field": "mlp_layer_types",
            "value": values.mlp_layer_types,
            "rule": "num_hash_layers=3 → 前 3 项 hash_moe + 其余 moe",
            "source_refs": [
                cfg("num_hash_layers", "legacy 字段"),
                rule("264-268", "__post_init__ mlp_layer_types 派生"),
            ],
        },
        {
            "field": "quantization_config",
            "value": values.quantization_config,
            "rule": "根配置声明 FP8；本样板只走 HF 逻辑 eager 路径，不分析量化执行（明确未覆盖）",
            "source_refs": [cfg("quantization_config", "根配置字段")],
        },
    ]
    return records
