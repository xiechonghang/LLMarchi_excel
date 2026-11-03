"""GLM-5.3 官方 HF 实现的有效配置解释（静态只读，不实例化/不运行模型）。

身份（2026-10-28 锁定）：
- 官方模型仓库 ``zai-org/GLM-5.3``（Hugging Face），revision
  ``ba32f8c19c8738e2c57c71798c601d477a7bf844``；根 ``config.json`` 的
  ``model_type="glm_moe_dsa"``、``architectures=["GlmMoeDsaForCausalLM"]``。
  该仓库 ``transformers_version="5.15.0"``，即保存 checkpoint 时的 transformers 版本。
- 实现源码：工作区本地官方 transformers 检出 ``hflib/transformers``（HEAD 见
  ``source_revision``，本检出为 5.19.0.dev0），模块
  ``src/transformers/models/glm_moe_dsa/``。模型卡指向的正是该模块
  （docs/source/en/model_doc/glm_moe_dsa.md）；本票以本地检出为该实现的固定版本。
- 本文件只读 ``config.json`` 与本地源码，按
  ``configuration_glm_moe_dsa.py::GlmMoeDsaConfig.__post_init__``（:135-167）与
  继承的 ``PreTrainedConfig`` 标准化规则推导实例化后实际生效的值；不导入
  torch/transformers、不实例化配置类、不运行模型、不下载权重。

关键解释事实（每条以源码行区间为据；行号按本地检出 536ecc00…）：
- ``layer_types`` 全部为 ``"indexed_attention"``（configuration :158-160）；这是所有
  78 个 decoder layer 的 Cache 类分派依据（cache_utils :1290
  ``"indexed_attention": DynamicIndexedLayer``），不是滑动窗口/压缩注意力。
- ``indexer_types`` 逐层 ``"full"``/``"shared"``（configuration :136-148）：``shared``
  层不实例化 indexer（modeling :348-349），复用上一 full 层的 top-k 选择
  （modeling :404-417，``prev_topk_indices``）。本配置显式给出 78 项，且与
  ``index_topk_freq=4``/``index_skip_topk_offset=3`` 的推导一致（本模块复核）。
- ``qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192+64 = 256``
  （configuration :149）；``head_dim`` 被 ``__post_init__`` 改写为
  ``qk_rope_head_dim=64``（configuration :150-152）——根 config.json 里的
  ``"head_dim": 192`` 是保存时的字段，HF 旋转位置编码实际只用 64 维（32 对）：
  ``RotaryEmbedding.compute_default_rope_parameters`` 读 ``config.head_dim``
  （modeling :96），``apply_rotary_pos_emb_interleave`` 只取 cos/sin 的前一半
  （modeling :142-149）。
- ``mlp_layer_types``：前 ``first_k_dense_replace=3`` 层 dense（intermediate_size
  =12288），其余 sparse（MoE：n_routed_experts=256、moe_intermediate_size=2048、
  n_shared_experts=1、num_experts_per_tok=8、routed_scaling_factor=2.5、
  n_group=1/topk_group=1、scoring_func=sigmoid、norm_topk_prob=true）
  （configuration :153-157；modeling :470-572）。
- ``num_nextn_predict_layers=1`` 是训练期 MTP 头：本地实现
  ``_keys_to_ignore_on_load_unexpected = [r"model\\.layers\\.78.*"]``
  （modeling :638）明确忽略 checkpoint 的 layer 78；``GlmMoeDsaModel.forward``
  （:712-722）只循环 ``layers[:num_hidden_layers]``（0..77）。MTP 不在本票
  “token 输入到 logits 的语言主干”内，明确标为未分析。
- 注意力缩放：``self.scaling = yarn_apply_mscale(config.rope_parameters,
  qk_head_dim ** -0.5)``（modeling :346）；本配置 ``rope_type="default"``，函数
  直接返回原缩放（:288-295），即 256^-0.5 = 1/16，无 YaRN mscale。
- 缓存：``past_key_values.update(k_pass, k_rot, layer_idx)``（modeling :396-398）
  写入的是**压缩后的 latent**：k_pass [B,1,S,512] 进 keys、k_rot [B,1,S,64] 进
  values（``DynamicLayer.update`` cache_utils :132-153）；使用前才
  ``kv_b_proj`` 展开全部 Lkv 个位置（modeling :402/:351-368）。indexer 另有
  ``update_indexer`` 的 [B,Lidx,128] 键缓存（cache_utils :350-365），仅 full 层有。
- ``quantization_config`` 声明 FP8（e4m3, block 128×128）：本票只分析 HF 逻辑
  eager 路径，不分析量化 kernel/数值，明确列为未覆盖。
- 视觉/图像输入路径不在本票范围：GLM-5.3 本身是 text-generation 仓库，
  ``pipeline_tag="text-generation"``；本票不读取/解释任何图像编码器。

``SOURCE_FILES`` 是所选 eager forward 路径实际读取的定义闭包（相对工作区根）；
``ANALYZED_SOURCE_FINGERPRINTS`` 固定已分析版本，来源漂移即具名拒绝。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "HfConfigError",
    "HfModelValues",
    "SOURCE_FILES",
    "MODEL_REPOSITORY",
    "ANALYZED_SOURCE_FINGERPRINTS",
    "LAYER_GROUP_SPECS",
    "verify_source_fingerprints",
    "load_hf_model_values",
    "layer_group_assignments",
    "effective_parameter_records",
    "effective_indexer_types",
    "effective_mlp_layer_types",
    "effective_layer_types",
    "derive_indexer_types",
    "GLM53_SCENARIOS",
]

# GLM-5.3 的少量有区别场景（不是长度组合枚举）：本模型没有滑动窗口/压缩窗口，
# 因此场景只检验「空缓存 / 非零历史 / 多 token / 超过 index_topk 的历史」四类
# 语义差异；窗口闭合这类 DeepSeek 概念不套用。
GLM53_SCENARIOS = (
    {
        "id": "prefill-empty",
        "label": "空缓存 Prefill（示例 seqlen=8）",
        "seqlen": 8,
        "kvcachelen": 0,
        "note": "kvcachelen=0 即空缓存起步；seqlen=8 只是示例取值，结构 shape 仍保留符号 seqlen。"
                "Lidx≤index_topk 时 top-k 全选（稀疏 mask 不删位置）。",
    },
    {
        "id": "prefill-symbolic",
        "label": "空缓存 Prefill（符号 seqlen，未给数值）",
        "seqlen": None,
        "kvcachelen": 0,
        "note": "seqlen 保持符号：与 seqlen 有关的派生量如实标为未知，不默认成 0。",
    },
    {
        "id": "decode-nonzero",
        "label": "非零历史单 token Decode（kvcachelen=4096）",
        "seqlen": 1,
        "kvcachelen": 4096,
        "note": "kvcachelen=4096 是示例逻辑历史长度；主 KV 与 indexer 缓存都线性保留，"
                "Lkv=Lidx=4097，top-k 截到 2048。",
    },
    {
        "id": "decode-long",
        "label": "长历史单 token Decode（kvcachelen=131072）",
        "seqlen": 1,
        "kvcachelen": 131072,
        "note": "历史远超 index_topk=2048：K_sel 饱和为 2048，剩余 key 被稀疏 mask 屏蔽；"
                "Lkv 仍为全部历史+新 token（无压缩条目）。",
    },
    {
        "id": "decode-multi",
        "label": "非零历史多 token Decode（seqlen=4，kvcachelen=4092）",
        "seqlen": 4,
        "kvcachelen": 4092,
        "note": "多 token Decode：检查统一结构按 seqlen 表达，而不是固定 +1/T=1。",
    },
)

# 官方模型仓库身份（HF API /api/models/zai-org/GLM-5.3 于 2026-10-28 返回的 sha）。
MODEL_REPOSITORY = {
    "repo_id": "zai-org/GLM-5.3",
    "revision": "ba32f8c19c8738e2c57c71798c601d477a7bf844",
    "last_modified": "2026-10-14T12:22:48.000Z",
    "pipeline_tag": "text-generation",
    "library_name": "transformers",
    "checkpoint_transformers_version": "5.15.0",
    "note": (
        "配置与模型卡按上述 revision 取得；只取 config.json 与 README.md（小文件），"
        "不下载 141 个 safetensors 权重分片。"
    ),
}

# 所选 eager forward 路径实际读取的定义闭包（相对工作区根）。
SOURCE_FILES = {
    "config.json": "models/zai-org/GLM-5.3/config.json",
    "modeling_glm_moe_dsa.py": (
        "hflib/transformers/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py"
    ),
    "configuration_glm_moe_dsa.py": (
        "hflib/transformers/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py"
    ),
    "cache_utils.py": "hflib/transformers/src/transformers/cache_utils.py",
    "masking_utils.py": "hflib/transformers/src/transformers/masking_utils.py",
    "modeling_rope_utils.py": "hflib/transformers/src/transformers/modeling_rope_utils.py",
    "activations.py": "hflib/transformers/src/transformers/activations.py",
}

# 已分析来源指纹（SHA256，本地检出 536ecc00… / 官方 config revision 见上）。
# 来源漂移（任一文件变化）时构建具名拒绝、要求重新分析后更新指纹，而不是拿旧
# 结论继续产出。
ANALYZED_SOURCE_FINGERPRINTS = {
    "config.json": "3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69",
    "modeling_glm_moe_dsa.py": "68a88d0d9097b7e37dab99999611d1b5ed57d3632add66c7ab26446f354e6be2",
    "configuration_glm_moe_dsa.py": "24cf543686badb3b41881d8a1ff556a5ee5d966efa2886ebbebdcaa378de75e1",
    "cache_utils.py": "2bef552bc39e8a4dc7ddfcf25740e6a4a7c4b836d92f5e56629a8ebfcd9cb246",
    "masking_utils.py": "c1d0e1e61754db819382bb03acdd2ab38c1bed3e9e9cf010918d2bf2771a8415",
    "modeling_rope_utils.py": "6a0727073e080e163d0287e15fc1acb90035e9a9df9cb19a13b632a40abfac24",
    "activations.py": "5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665",
}


def verify_source_fingerprints(workspace_root: Path) -> None:
    """复算 SOURCE_FILES 各输入 SHA256 并与已分析指纹核对；漂移具名拒绝。"""
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
    if not isinstance(raw, dict):
        raise HfConfigError(f"HF 根配置顶层必须是对象: {path}")
    if raw.get("model_type") != "glm_moe_dsa":
        raise HfConfigError(
            f"HF 根配置 model_type 必须是 glm_moe_dsa（GLM-5.3 官方实现）: {path}"
        )
    if raw.get("architectures") != ["GlmMoeDsaForCausalLM"]:
        raise HfConfigError(
            f"HF 根配置 architectures 必须是 ['GlmMoeDsaForCausalLM']: {path}"
        )
    return raw


def derive_indexer_types(freq: int, offset: int, num_layers: int) -> list[str]:
    """``indexer_types`` 的源码推导（configuration :143-148）：freq/offset 调度。

    ``full`` 当 ``max(i-offset+1, 0) % freq == 0``，否则 ``shared``。仅在根配置
    未显式提供 ``indexer_types`` 时作为来源；显式列表与推导不一致时本模块拒绝
    （不静默选一边）。
    """
    freq = max(int(freq), 1)
    return [
        "full" if (max(i - int(offset) + 1, 0) % freq) == 0 else "shared"
        for i in range(int(num_layers))
    ]


def effective_indexer_types(raw: dict) -> list[str]:
    """逐层 indexer 模式：显式列表 > index_topk_pattern > freq/offset 推导。"""
    n = raw["num_hidden_layers"]
    explicit = raw.get("indexer_types")
    pattern = raw.get("index_topk_pattern")
    freq = raw.get("index_topk_freq")
    offset = raw.get("index_skip_topk_offset")
    derived = None
    if pattern is not None:
        if not isinstance(pattern, str):
            raise HfConfigError(f"index_topk_pattern 只分析字符串模式: {pattern!r}")
        mapped = []
        for index, char in enumerate(pattern):
            if char not in ("F", "S"):
                raise HfConfigError(
                    f"index_topk_pattern[{index}]={char!r} 不在 'F'/'S'（full/shared）内"
                )
            mapped.append("full" if char == "F" else "shared")
        derived = mapped
    elif freq is not None or offset is not None:
        derived = derive_indexer_types(
            freq if freq is not None else 1,
            offset if offset is not None else 2,
            n,
        )
    if explicit is not None:
        if not isinstance(explicit, list) or len(explicit) != n:
            raise HfConfigError(
                f"indexer_types 必须是长度 {n} 的数组（实得 {explicit!r}）"
            )
        for index, value in enumerate(explicit):
            if value not in ("full", "shared"):
                raise HfConfigError(
                    f"indexer_types[{index}]={value!r} 不在 'full'/'shared' 内"
                )
        if derived is not None and list(explicit) != derived:
            raise HfConfigError(
                "根配置的 indexer_types 与 index_topk_pattern/freq+offset 推导不一致；"
                "不静默选择其中一边"
            )
        return list(explicit)
    if derived is None:
        raise HfConfigError(
            "根配置缺少 indexer_types，且没有 index_topk_pattern/freq/offset 可推导"
        )
    if len(derived) != n:
        raise HfConfigError(
            f"推导的 indexer_types 长度 {len(derived)} 与 num_hidden_layers={n} 不一致"
        )
    return derived


def effective_mlp_layer_types(raw: dict) -> list[str]:
    """MLP 层型：显式列表 > first_k_dense_replace 推导（configuration :153-157）。"""
    n = raw["num_hidden_layers"]
    explicit = raw.get("mlp_layer_types")
    if explicit is not None:
        if not isinstance(explicit, list) or len(explicit) != n:
            raise HfConfigError(
                f"mlp_layer_types 必须是长度 {n} 的数组（实得 {explicit!r}）"
            )
        for index, value in enumerate(explicit):
            if value not in ("dense", "sparse"):
                raise HfConfigError(
                    f"mlp_layer_types[{index}]={value!r} 不在 'dense'/'sparse' 内"
                )
        return list(explicit)
    n_dense = int(raw.get("first_k_dense_replace", 3))
    n_dense = min(n_dense, n)
    return ["dense"] * n_dense + ["sparse"] * (n - n_dense)


def effective_layer_types(raw: dict) -> list[str]:
    """注意力层型：所有层都是 ``indexed_attention``（configuration :158-160）。"""
    n = raw["num_hidden_layers"]
    explicit = raw.get("layer_types")
    if explicit is not None:
        if not isinstance(explicit, list) or len(explicit) != n:
            raise HfConfigError(
                f"layer_types 必须是长度 {n} 的数组（实得 {explicit!r}）"
            )
        unexpected = [value for value in explicit if value != "indexed_attention"]
        if unexpected:
            raise HfConfigError(
                "本地 glm_moe_dsa 实现只分派 indexed_attention 层；根配置出现其它层型: "
                f"{sorted(set(unexpected))}"
            )
        return list(explicit)
    return ["indexed_attention"] * n


@dataclass(frozen=True)
class HfModelValues:
    """layer_idx 视角的 GLM-5.3 HF 有效配置值（静态解释，非运行时探值）。"""

    # 身份
    model_key: str
    display_name: str
    implementation: str
    source_revision: str
    config_path: str
    model_revision: str
    checkpoint_transformers_version: str
    # 全局
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    config_head_dim_raw: int
    vocab_size: int
    rms_norm_eps: float
    hidden_act: str
    use_cache: bool
    max_position_embeddings: int
    tie_word_embeddings: bool
    # MLA 注意力
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    attention_bias: bool
    attention_dropout: float
    attention_scaling: float
    # DSA indexer
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    indexer_types: list[str]
    index_topk_freq: int
    index_skip_topk_offset: int
    # RoPE
    rope_theta: float
    rope_type: str
    # MoE
    first_k_dense_replace: int
    intermediate_size: int
    moe_intermediate_size: int
    n_shared_experts: int
    n_routed_experts: int
    num_experts_per_tok: int
    scoring_func: str
    norm_topk_prob: bool
    routed_scaling_factor: float
    n_group: int
    topk_group: int
    moe_router_dtype: str
    # 层规划
    layer_types: list[str]
    mlp_layer_types: list[str]
    # 声明但明确不在本票范围
    quantization_config: dict
    num_nextn_predict_layers: int

    def layer_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def mlp_layer_type(self, layer_idx: int) -> str:
        return self.mlp_layer_types[layer_idx]

    def indexer_type(self, layer_idx: int) -> str:
        return self.indexer_types[layer_idx]


def load_hf_model_values(
    workspace_root: str | Path,
    model_key: str = "glm_5_3_hf",
) -> HfModelValues:
    """读取根 config.json 并静态解释为 GLM-5.3 HF 有效配置；失配/缺字段具名拒绝。"""
    workspace_root = Path(workspace_root)
    raw = _load_root_config(workspace_root)

    def _require(name: str):
        value = raw.get(name)
        if value is None:
            raise HfConfigError(f"根配置缺少必要字段 {name!r}")
        return value

    layer_types = effective_layer_types(raw)
    mlp_types = effective_mlp_layer_types(raw)
    indexer_types = effective_indexer_types(raw)

    qk_nope = int(_require("qk_nope_head_dim"))
    qk_rope = int(_require("qk_rope_head_dim"))
    qk_head_dim = qk_nope + qk_rope
    declared_qk = raw.get("qk_head_dim")
    if declared_qk is not None and int(declared_qk) != qk_head_dim:
        raise HfConfigError(
            f"根配置 qk_head_dim={declared_qk} 与 qk_nope+qk_rope={qk_head_dim} 不一致"
        )
    config_head_dim_raw = int(_require("head_dim"))
    # __post_init__ :150-152 把 head_dim 指向 rope slice；config 的 192 不参与 RoPE。
    rope_parameters = raw.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        raise HfConfigError("根配置缺少 rope_parameters 对象（需 rope_type/rope_theta）")
    rope_type = rope_parameters.get("rope_type", "default")
    rope_theta = rope_parameters.get("rope_theta")
    if rope_theta is None:
        raise HfConfigError("根配置 rope_parameters 缺少 rope_theta")

    checkout_root = workspace_root / "hflib" / "transformers"
    source_revision = _git_head(checkout_root)

    return HfModelValues(
        model_key=model_key,
        display_name="GLM-5.3 (zai-org/GLM-5.3, HF transformers glm_moe_dsa)",
        implementation="transformers",
        source_revision=source_revision,
        config_path=SOURCE_FILES["config.json"],
        model_revision=MODEL_REPOSITORY["revision"],
        checkpoint_transformers_version=MODEL_REPOSITORY["checkpoint_transformers_version"],
        hidden_size=int(_require("hidden_size")),
        num_hidden_layers=int(_require("num_hidden_layers")),
        num_attention_heads=int(_require("num_attention_heads")),
        num_key_value_heads=int(_require("num_key_value_heads")),
        head_dim=qk_rope,  # __post_init__ 后的实际 RoPE 维
        config_head_dim_raw=config_head_dim_raw,
        vocab_size=int(_require("vocab_size")),
        rms_norm_eps=float(_require("rms_norm_eps")),
        hidden_act=raw.get("hidden_act", "silu"),
        use_cache=bool(raw.get("use_cache", True)),
        max_position_embeddings=int(_require("max_position_embeddings")),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        q_lora_rank=int(_require("q_lora_rank")),
        kv_lora_rank=int(_require("kv_lora_rank")),
        qk_head_dim=qk_head_dim,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=int(_require("v_head_dim")),
        attention_bias=bool(raw.get("attention_bias", False)),
        attention_dropout=float(raw.get("attention_dropout", 0.0)),
        attention_scaling=qk_head_dim ** -0.5,  # yarn_apply_mscale(default) = 原值
        index_n_heads=int(_require("index_n_heads")),
        index_head_dim=int(_require("index_head_dim")),
        index_topk=int(_require("index_topk")),
        indexer_types=indexer_types,
        index_topk_freq=int(raw.get("index_topk_freq", 1)),
        index_skip_topk_offset=int(raw.get("index_skip_topk_offset", 2)),
        rope_theta=float(rope_theta),
        rope_type=str(rope_type),
        first_k_dense_replace=int(raw.get("first_k_dense_replace", 3)),
        intermediate_size=int(_require("intermediate_size")),
        moe_intermediate_size=int(_require("moe_intermediate_size")),
        n_shared_experts=int(raw.get("n_shared_experts", 1)),
        n_routed_experts=int(_require("n_routed_experts")),
        num_experts_per_tok=int(_require("num_experts_per_tok")),
        scoring_func=str(raw.get("scoring_func", "sigmoid")),
        norm_topk_prob=bool(raw.get("norm_topk_prob", True)),
        routed_scaling_factor=float(_require("routed_scaling_factor")),
        n_group=int(raw.get("n_group", 1)),
        topk_group=int(raw.get("topk_group", 1)),
        moe_router_dtype=str(raw.get("moe_router_dtype", "float32")),
        layer_types=layer_types,
        mlp_layer_types=mlp_types,
        quantization_config=raw.get("quantization_config", {}),
        num_nextn_predict_layers=int(raw.get("num_nextn_predict_layers", 0)),
    )


def _git_head(checkout_root: Path) -> str:
    """本地 Transformers 检出的 HEAD 身份；不可得时如实标注 unknown。"""
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
    """逐参数 raw 字段/默认/派生规则与来源（供参数符号区联查）。"""
    modeling = SOURCE_FILES["modeling_glm_moe_dsa.py"]
    configuration = SOURCE_FILES["configuration_glm_moe_dsa.py"]
    config_json = SOURCE_FILES["config.json"]
    rope_utils = SOURCE_FILES["modeling_rope_utils.py"]
    activations = SOURCE_FILES["activations.py"]
    cache_utils = SOURCE_FILES["cache_utils.py"]
    masking = SOURCE_FILES["masking_utils.py"]

    def cfg(locator: str, note: str) -> dict:
        return {"path": config_json, "locator": locator, "kind": "config", "note": note}

    def rule(locator: str, note: str) -> dict:
        return {"path": configuration, "locator": locator, "kind": "derived", "note": note}

    def src(path: str, locator: str, note: str) -> dict:
        return {"path": path, "locator": locator, "kind": "source", "note": note}

    return [
        {
            "field": "model_identity",
            "value": {
                "repo_id": MODEL_REPOSITORY["repo_id"],
                "revision": MODEL_REPOSITORY["revision"],
                "model_type": "glm_moe_dsa",
                "architectures": ["GlmMoeDsaForCausalLM"],
            },
            "rule": "官方模型仓库 revision 与根配置 model_type/architectures 双条件锁定；"
            "不以目录名或相似模型（GLM-5/GLM-5.3-Flash）代替",
            "source_refs": [
                cfg("model_type", "根配置字段"),
                cfg("architectures", "根配置字段"),
            ],
        },
        {
            "field": "hidden_size",
            "value": values.hidden_size,
            "rule": "根配置直接字段",
            "source_refs": [cfg("hidden_size", "根配置字段"), rule("93", "配置类字段声明")],
        },
        {
            "field": "num_hidden_layers",
            "value": values.num_hidden_layers,
            "rule": "根配置直接字段；主干层数 78（0..77）；checkpoint 的 layer 78 是 MTP，"
            "本地实现显式忽略（modeling :638），不进入本主干",
            "source_refs": [
                cfg("num_hidden_layers", "根配置字段"),
                rule("96", "配置类字段声明"),
                src(modeling, "638", "_keys_to_ignore_on_load_unexpected = [r\"model\\.layers\\.78.*\"]"),
                src(modeling, "712-722", "for i, decoder_layer in enumerate(self.layers[:num_hidden_layers])"),
            ],
        },
        {
            "field": "num_attention_heads",
            "value": values.num_attention_heads,
            "rule": "根配置直接字段；64 个 query 头，KV 头同为 64（num_key_value_groups=1）",
            "source_refs": [
                cfg("num_attention_heads", "根配置字段"),
                src(modeling, "321", "self.num_key_value_groups = num_attention_heads // num_key_value_heads"),
            ],
        },
        {
            "field": "num_key_value_heads",
            "value": values.num_key_value_heads,
            "rule": "根配置直接字段；与 query 头数相同，repeat_kv 为恒等（不复制 KV 头）",
            "source_refs": [cfg("num_key_value_heads", "根配置字段"),
                            src(modeling, "245-254", "repeat_kv：n_rep==1 直接返回")],
        },
        {
            "field": "qk_head_dim",
            "value": values.qk_head_dim,
            "rule": "派生 = qk_nope_head_dim + qk_rope_head_dim = 192+64",
            "source_refs": [rule("149", "__post_init__ qk_head_dim 派生")],
        },
        {
            "field": "qk_nope_head_dim",
            "value": values.qk_nope_head_dim,
            "rule": "根配置直接字段；不旋转的 key/query 维",
            "source_refs": [cfg("qk_nope_head_dim", "根配置字段"), rule("107", "配置类字段声明")],
        },
        {
            "field": "qk_rope_head_dim",
            "value": values.qk_rope_head_dim,
            "rule": "根配置直接字段；同时是 __post_init__ 后的 head_dim（RoPE 只用这 64 维）",
            "source_refs": [
                cfg("qk_rope_head_dim", "根配置字段"),
                rule("150-152", "__post_init__: self.head_dim = self.qk_rope_head_dim"),
                src(modeling, "142-149", "cos/sin 取前一半，按 (0::2,1::2) 交错应用"),
            ],
        },
        {
            "field": "head_dim",
            "value": values.head_dim,
            "rule": "运行期值 = qk_rope_head_dim=64；根配置的 head_dim=192 在 __post_init__ "
            "被覆盖，不参与 RoPE（若按 192 解释会得到错误的频率数）",
            "source_refs": [
                cfg("head_dim", "根配置保存值（被覆盖）"),
                rule("150-152", "__post_init__ 覆盖规则"),
                src(rope_utils, "84-101", "RotaryEmbedding.compute_default_rope_parameters 读 config.head_dim"),
            ],
        },
        {
            "field": "v_head_dim",
            "value": values.v_head_dim,
            "rule": "根配置直接字段；value 头维（与 qk_nope 不同，k/v 展开后分别 192/256）",
            "source_refs": [cfg("v_head_dim", "根配置字段"),
                            src(modeling, "334-338", "kv_b_proj 输出 num_heads*(192+256)")],
        },
        {
            "field": "q_lora_rank",
            "value": values.q_lora_rank,
            "rule": "根配置直接字段；q_a_proj 低秩输出（q_a_layernorm 的 RMSNorm 维）",
            "source_refs": [cfg("q_lora_rank", "根配置字段"),
                            src(modeling, "324-326", "q_a_proj → q_a_layernorm → q_b_proj")],
        },
        {
            "field": "kv_lora_rank",
            "value": values.kv_lora_rank,
            "rule": "根配置直接字段；压缩 KV latent 维（缓存 k_pass 的最后一维）",
            "source_refs": [cfg("kv_lora_rank", "根配置字段"),
                            src(modeling, "328-338", "kv_a_proj_with_mqa 输出 512+64"),
                            src(modeling, "396-398", "缓存写入压缩 latent")],
        },
        {
            "field": "vocab_size",
            "value": values.vocab_size,
            "rule": "根配置直接字段；embedding 与 lm_head 的词表规模",
            "source_refs": [cfg("vocab_size", "根配置字段"), rule("92", "配置类字段声明")],
        },
        {
            "field": "rms_norm_eps",
            "value": values.rms_norm_eps,
            "rule": "根配置直接字段；input/post_attention/final RMSNorm 的 epsilon。"
            "q_a_layernorm/kv_a_layernorm 与 indexer k_norm 用源码默认 1e-6（不同 eps）",
            "source_refs": [
                cfg("rms_norm_eps", "根配置字段"),
                src(modeling, "325", "GlmMoeDsaRMSNorm(self.q_lora_rank)（默认 eps=1e-6）"),
                src(modeling, "333", "GlmMoeDsaRMSNorm(config.kv_lora_rank)（默认 eps=1e-6）"),
                src(modeling, "180", "nn.LayerNorm(self.head_dim, eps=1e-6)（indexer k_norm）"),
            ],
        },
        {
            "field": "hidden_act",
            "value": values.hidden_act,
            "rule": "根配置直接字段；SiLU（共享专家/稠密 MLP/专家）",
            "source_refs": [cfg("hidden_act", "根配置字段"),
                            src(activations, "ACT2FN['silu']", "SiLUActivation")],
        },
        {
            "field": "use_cache",
            "value": values.use_cache,
            "rule": "根配置直接字段；本分析启用 KV 缓存",
            "source_refs": [cfg("use_cache", "根配置字段"),
                            src(modeling, "688-689", "use_cache 且未传入时新建 DynamicCache")],
        },
        {
            "field": "max_position_embeddings",
            "value": values.max_position_embeddings,
            "rule": "根配置直接字段；RoPE 缓冲的 max_seq_len_cached（不改变前向结构）",
            "source_refs": [cfg("max_position_embeddings", "根配置字段"),
                            rule("113", "配置类字段声明")],
        },
        {
            "field": "rope_parameters",
            "value": {"rope_type": values.rope_type, "rope_theta": values.rope_theta,
                      "head_dim": values.head_dim},
            "rule": "根配置 rope_parameters；default 型、theta=8e6、只用 64 维（32 对频率）",
            "source_refs": [
                cfg("rope_parameters", "根配置字段"),
                src(modeling, "84-101", "compute_default_rope_parameters"),
                src(modeling, "103-111", "cos/sin 计算"),
                src(rope_utils, "ROPE_INIT_FUNCTIONS", "default 分支"),
            ],
        },
        {
            "field": "attention_scaling",
            "value": values.attention_scaling,
            "rule": "qk_head_dim**-0.5 = 256^-0.5 = 1/16；rope_type=default 时 "
            "yarn_apply_mscale 不乘 mscale",
            "source_refs": [
                src(modeling, "346", "self.scaling = yarn_apply_mscale(rope_parameters, qk_head_dim**-0.5)"),
                src(modeling, "288-295", "yarn_apply_mscale：default 直接返回 scaling"),
            ],
        },
        {
            "field": "attention_bias",
            "value": values.attention_bias,
            "rule": "根配置直接字段；q_a_proj/kv_a_proj_with_mqa/o_proj 的 bias=False",
            "source_refs": [cfg("attention_bias", "根配置字段"),
                            src(modeling, "324-344", "Linear(..., bias=config.attention_bias)")],
        },
        {
            "field": "index_n_heads",
            "value": values.index_n_heads,
            "rule": "根配置直接字段；DSA indexer 的 32 个打分头",
            "source_refs": [cfg("index_n_heads", "根配置字段"),
                            src(modeling, "172-181", "Indexer 的 wq_b/wk/weights_proj 形状")],
        },
        {
            "field": "index_head_dim",
            "value": values.index_head_dim,
            "rule": "根配置直接字段；indexer 头维 128（k_norm 是带 bias 的 LayerNorm）",
            "source_refs": [cfg("index_head_dim", "根配置字段"),
                            src(modeling, "180", "k_norm = nn.LayerNorm(head_dim, eps=1e-6)")],
        },
        {
            "field": "index_topk",
            "value": values.index_topk,
            "rule": "根配置直接字段；每 query 选择的 key 数上限 2048（实际取 "
            "min(index_topk, Lidx)）",
            "source_refs": [cfg("index_topk", "根配置字段"),
                            src(modeling, "241-242", "topk = min(self.index_topk, index_scores.shape[-1])")],
        },
        {
            "field": "indexer_types",
            "value": values.indexer_types,
            "rule": "显式 78 项；与 index_topk_freq=4/index_skip_topk_offset=3 的推导一致。"
            "shared 层不实例化 indexer，复用上一 full 层的 top-k",
            "source_refs": [
                cfg("indexer_types", "根配置显式数组"),
                cfg("index_topk_freq", "根配置字段"),
                cfg("index_skip_topk_offset", "根配置字段"),
                rule("136-148", "__post_init__ indexer_types 推导"),
                src(modeling, "348-349", "skip_topk → indexer=None"),
                src(modeling, "404-417", "shared 层消费 prev_topk_indices"),
            ],
        },
        {
            "field": "mlp_layer_types",
            "value": values.mlp_layer_types,
            "rule": "显式 78 项：前 3 层 dense，其余 sparse（= first_k_dense_replace=3）",
            "source_refs": [
                cfg("mlp_layer_types", "根配置显式数组"),
                cfg("first_k_dense_replace", "根配置字段"),
                rule("153-157", "__post_init__ mlp_layer_types 派生"),
                src(modeling, "581", "sparse → GlmMoeDsaMoE，否则 GlmMoeDsaMLP"),
            ],
        },
        {
            "field": "intermediate_size",
            "value": values.intermediate_size,
            "rule": "根配置直接字段；dense 层 MLP 的中间维 12288",
            "source_refs": [cfg("intermediate_size", "根配置字段"),
                            src(modeling, "455-463", "GlmMoeDsaMLP 用 config.intermediate_size")],
        },
        {
            "field": "moe_intermediate_size",
            "value": values.moe_intermediate_size,
            "rule": "根配置直接字段；单专家中间维 2048（共享专家 = 2048×n_shared_experts）",
            "source_refs": [cfg("moe_intermediate_size", "根配置字段"),
                            src(modeling, "519-521", "gate_up_proj [E, 2*Es, H]；down_proj [E, H, Es]"),
                            src(modeling, "561-563", "shared_experts 中间维 = Es*n_shared_experts")],
        },
        {
            "field": "n_routed_experts",
            "value": values.n_routed_experts,
            "rule": "根配置字段 n_routed_experts；标准名 num_local_experts 经 attribute_map "
            "映射（configuration :90）",
            "source_refs": [
                cfg("n_routed_experts", "根配置字段"),
                rule("90", "attribute_map num_local_experts → n_routed_experts"),
                src(modeling, "474", "self.num_experts = config.num_local_experts"),
            ],
        },
        {
            "field": "n_shared_experts",
            "value": values.n_shared_experts,
            "rule": "根配置直接字段；1 个共享专家（宽度 2048），与 routed 专家并行",
            "source_refs": [cfg("n_shared_experts", "根配置字段"),
                            src(modeling, "561-563", "self.shared_experts = GlmMoeDsaMLP(...)"),
                            src(modeling, "571", "routed + shared_experts(residuals)")],
        },
        {
            "field": "num_experts_per_tok",
            "value": values.num_experts_per_tok,
            "rule": "根配置直接字段；每 token 选 8 个专家（与专家总数 256 不同参数）",
            "source_refs": [cfg("num_experts_per_tok", "根配置字段"),
                            src(modeling, "502", "topk(scores_for_choice, k=self.top_k)")],
        },
        {
            "field": "scoring_func",
            "value": values.scoring_func,
            "rule": "根配置直接字段；sigmoid 打分（不是 softmax，也不是 DeepSeek 的 "
            "sqrt(softplus)）",
            "source_refs": [cfg("scoring_func", "根配置字段"),
                            src(modeling, "486", "scores = router_logits.sigmoid()")],
        },
        {
            "field": "norm_topk_prob",
            "value": values.norm_topk_prob,
            "rule": "根配置直接字段；TopK 权重按和归一（+1e-20）",
            "source_refs": [cfg("norm_topk_prob", "根配置字段"),
                            src(modeling, "504-506", "topk_weights /= topk_weights.sum(-1,keepdim)+1e-20")],
        },
        {
            "field": "routed_scaling_factor",
            "value": values.routed_scaling_factor,
            "rule": "根配置直接字段；归一后乘 2.5",
            "source_refs": [cfg("routed_scaling_factor", "根配置字段"),
                            src(modeling, "507", "topk_weights = topk_weights * routed_scaling_factor")],
        },
        {
            "field": "n_group",
            "value": values.n_group,
            "rule": "根配置直接字段；n_group=1、topk_group=1：源码仍执行 group_scores/"
            "group_mask，但单组下掩码全通过，不改变选择集合（如实表达，不省略也不夸大）",
            "source_refs": [cfg("n_group", "根配置字段"),
                            cfg("topk_group", "根配置字段"),
                            src(modeling, "488-501", "group_scores → group_idx → score_mask → masked_fill")],
        },
        {
            "field": "moe_router_dtype",
            "value": values.moe_router_dtype,
            "rule": "根配置声明 float32；本地实现固定把 hidden/weight 转 float32 计算 "
            "（不读该字段），因此路由精度事实来自源码而非配置",
            "source_refs": [cfg("moe_router_dtype", "根配置字段"),
                            src(modeling, "485", "F.linear(hidden_states.type(float32), weight.type(float32))")],
        },
        {
            "field": "quantization_config",
            "value": values.quantization_config,
            "rule": "根配置声明 FP8（e4m3，block 128×128）；本票只分析 HF 逻辑 eager 路径，"
            "量化执行/kernel/数值等价明确未覆盖",
            "source_refs": [cfg("quantization_config", "根配置字段")],
        },
        {
            "field": "num_nextn_predict_layers",
            "value": values.num_nextn_predict_layers,
            "rule": "训练期 MTP 头数 1；本地 CausalLM 忽略 model.layers.78.* 权重，"
            "forward 不执行 MTP —— 不在本主干，未分析",
            "source_refs": [
                cfg("num_nextn_predict_layers", "根配置字段"),
                src(modeling, "638", "_keys_to_ignore_on_load_unexpected"),
                src(modeling, "712", "layers[:num_hidden_layers] 循环"),
            ],
        },
        {
            "field": "cache_dispatch",
            "value": {"layer_types": sorted(set(values.layer_types)),
                      "cache_class": "DynamicIndexedLayer"},
            "rule": "所有层 indexed_attention → DynamicIndexedLayer：主 KV 缓存 + indexer 键缓存",
            "source_refs": [
                src(cache_utils, "1290", "\"indexed_attention\": DynamicIndexedLayer"),
                src(cache_utils, "331-365", "DynamicIndexedLayer.update_indexer"),
                src(cache_utils, "132-153", "DynamicLayer.update：keys/values 追加"),
                src(modeling, "396-398", "缓存写入压缩 latent（512 + 64）"),
            ],
        },
        {
            "field": "mask_creation",
            "value": {"allow_is_causal_skip": False, "key": "indexed_attention"},
            "rule": "网络入口用 create_causal_mask(..., allow_is_causal_skip=False) 构建一次"
            "因果 mask，字典键 'indexed_attention'；层内再用 top-k 索引把它改成稀疏 mask",
            "source_refs": [
                src(modeling, "696-706", "causal_mask_mapping = {'indexed_attention': create_causal_mask(...)}"),
                src(masking, "850-", "create_causal_mask"),
                src(cache_utils, "165-172", "DynamicLayer.get_mask_sizes：kv_length=已缓存+query"),
            ],
        },
    ]


# ---------------------------------------------------------------------------
# 三组真实层组（indexer 模式 × MLP 类型）：归属、代表层与覆盖校验的单一事实来源
# ---------------------------------------------------------------------------

# 本配置里 dense 层（0..2）恰好都是 full indexer，因此只存在三种真实组合；
# 不虚构 "shared + dense" 组，也不把 78 层机械复制成 78 份模板。
LAYER_GROUP_SPECS = (
    {
        "id": "full-dense",
        "label": "Full DSA Indexer + Dense MLP",
        "attention_type": "indexed_attention",
        "indexer_type": "full",
        "moe_type": "dense",
        "representative_layer_index": 0,
    },
    {
        "id": "shared-moe",
        "label": "Shared DSA Indexer + TopK MoE",
        "attention_type": "indexed_attention",
        "indexer_type": "shared",
        "moe_type": "sparse",
        "representative_layer_index": 3,
    },
    {
        "id": "full-moe",
        "label": "Full DSA Indexer + TopK MoE",
        "attention_type": "indexed_attention",
        "indexer_type": "full",
        "moe_type": "sparse",
        "representative_layer_index": 6,
    },
)


def layer_group_assignments(values: HfModelValues) -> list[dict]:
    """按真实 indexer_types/mlp_layer_types 推导三组层归属（保持 SPECS 顺序）。

    返回 [{id,label,attention_type,indexer_type,moe_type,
    representative_layer_index,layer_indices,count}]。任何未归属组合、空组、
    代表层不在组内、或三组未恰好覆盖全部主干层，均具名拒绝（ValueError）。
    """
    by_type: dict[tuple[str, str], list[int]] = {}
    for index in range(values.num_hidden_layers):
        by_type.setdefault(
            (values.indexer_type(index), values.mlp_layer_type(index)), []
        ).append(index)
    groups: list[dict] = []
    covered: list[int] = []
    for spec in LAYER_GROUP_SPECS:
        layer_indices = by_type.pop(
            (spec["indexer_type"], spec["moe_type"]), []
        )
        if not layer_indices:
            raise ValueError(
                f"层组 {spec['id']} 在配置中没有真实层"
                f"（{spec['indexer_type']} + {spec['moe_type']}）"
            )
        representative = spec["representative_layer_index"]
        if representative not in layer_indices:
            raise ValueError(
                f"层组 {spec['id']} 代表层 {representative} 不属于真实层集合 {layer_indices}"
            )
        if values.layer_type(representative) != spec["attention_type"]:
            raise ValueError(
                f"层组 {spec['id']} 代表层 {representative} 的 attention 层型 "
                f"{values.layer_type(representative)!r} 与 {spec['attention_type']!r} 不符"
            )
        groups.append({
            **spec,
            "layer_indices": layer_indices,
            "count": len(layer_indices),
        })
        covered.extend(layer_indices)
    if by_type:
        leftover = {f"{idx_mode}+{mlp}": idx for (idx_mode, mlp), idx in by_type.items()}
        raise ValueError(f"存在未归属到任何已知层组的层型组合: {leftover}")
    expected = list(range(values.num_hidden_layers))
    if sorted(covered) != expected or len(covered) != len(set(covered)):
        raise ValueError(
            f"三组层集合未恰好覆盖 {values.num_hidden_layers} 层: 实得 {len(covered)} 项"
            f"（去重 {len(set(covered))}）"
        )
    return groups
