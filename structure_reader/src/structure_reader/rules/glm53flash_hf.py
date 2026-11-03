"""GLM-5.3-Flash 官方 HF 实现的有效配置解释（静态只读，不实例化/不运行模型）。

身份（2026-10-29 锁定）：
- 官方模型仓库 ``zai-org/GLM-5.3-Flash``（Hugging Face，``pipeline_tag=image-text-to-text``），
  revision ``46a050bec13002ad29ee795c16e61cf4fa3c4a9c``（2026-10-15）；根
  ``config.json`` 的 ``model_type="glm5_next"``、
  ``architectures=["Glm5NextForConditionalGeneration"]``，``text_config.model_type=
  "glm5_next_text"``。
- 实现源码：工作区本地官方 transformers 检出 ``hflib/transformers``（HEAD 见
  ``source_revision``，本检出为 5.19.0.dev0），模块
  ``src/transformers/models/glm5_next/``（configuration/modeling）。
- 本文件只读根 ``config.json`` 与本地源码，按
  ``Glm5NextConfig``/``Glm5NextTextConfig.__post_init__`` 与继承的
  ``PreTrainedConfig`` 标准化规则推导实例化后实际生效的值；不导入
  torch/transformers、不实例化配置类、不运行模型、不下载权重。

**这是 GLM-5.3-Flash，不是 GLM-5.3。** GLM-5.3 是 ``zai-org/GLM-5.3`` /
``glm_moe_dsa``（78 层、全 MLA+DSA、普通残差、无 mHC）；Flash 是
``glm5_next``：45 层混合 KDA 线性注意力（34 层）+ MLA/DSA 稀疏注意力（11 层）、
4 路 mHC 残差、NoPE（``qk_rope_head_dim=0``）、SwiGLU clamp、k-pool 压缩
DSA indexer、288 路由专家。两者**不能**共用注意力/缓存/残差结论。

关键解释事实（每条以源码行区间为据；行号按本地检出 536ecc00…）：

- ``layer_types``：34 个 ``linear_attention``（KDA：0,1,2,4,5,6,…44）与 11 个
  旧名 ``deepseek_sparse_attention``（3,7,11,…43）；``PreTrainedConfig.__post_init__``
  的 ``remap_legacy_layer_types`` 把它规范为 ``indexed_attention``
  （configuration_utils.py:87-134）。本地 ``Glm5NextTextModel.forward`` 用
  ``causal_mask_mapping[self.config.layer_types[i]]`` 取 2D padding mask
  （modeling:1495-1500），不是 4D 因果 mask。
- ``mlp_layer_types``：前 3 层 ``dense``（0/1/2），其余 42 层 ``sparse``（TopK MoE）。
- ``hc_mult=4``、``hc_sinkhorn_iters=20``、``hc_eps=1e-6``：每层双 mHC
  （``attn_hc``/``ffn_hc``）把 4 路流坍缩成单流、再把子层输出展开/混合回 4 路；
  **不是普通残差相加**（modeling:218-329/1278-1348）。网络入口把 embedding
  广播成 4 路相同流（modeling:1493），末端 ``hc_head`` 是 4 路**均值**
  （modeling:332-336/1509），与 DeepSeek 的加权 HyperHead 不同。
- KDA（Kimi Delta Attention，modeling:620-769）：q/k/v 投影 + depthwise 因果短卷积
  （kernel=4，含 conv state）+ forget gate（``-5·sigmoid(exp(A_log)·g)`` 下界）+
  beta + 分块/递推 delta-rule 扫描（状态 ``[B,64,128,128]``）+ gated RMSNorm +
  o_proj；状态大小固定，**不随历史长度增长**。
- MLA/DSA（modeling:1100-1275）：q 低秩（1536）+ NoPE（``qk_rope_head_dim=0``，
  K 只有 256 维 nope 段）；KV 缓存存 512 维压缩 latent（values 槽为 0 维 k_rot）；
  用时 ``kv_b_proj`` 展开全部 Lkv；indexer 用 k-pool=4 压缩池 + 每 query 选
  ``min(index_topk//kpool, 池数)`` 个池、展开成原始 token、再追当前不完整池的
  tail；稀疏 mask 由 selected counts scatter 得到（True=可见）。
- MoE（modeling:106-206）：sigmoid 打分 + ``e_score_correction_bias`` +
  288 选 8 + 归一(+1e-20) + ×2.5；专家 ``one_hot(top_k,E+1)``；专家与共享专家
  （宽度 2048）的 SwiGLU 都做 ``gate.clamp(max=10)`` / ``up.clamp(±10)``。
- MTP：``num_nextn_predict_layers=1``；本地实现
  ``_keys_to_ignore_on_load_unexpected=[r"layers\\.45\\.", r"layers\\.\\d+\\.shared_head\\."]``
  （modeling:1378）忽略 checkpoint 的 layer 45 与 shared_head；forward 只循环
  ``layers[:45]``（0..44）（modeling:1495-1496）——MTP 不在本主干。
- 视觉/图像输入路径不在本票范围：本文件只解释 text-only 语言主干；视觉
  ``Glm5NextVisionModel``（24 层）与图像/视频 embed 替换不分析。

版本差异（明确，不混用）：根配置记录 ``transformers_version="5.16.0"``，而本地
官方检出 HEAD 为 5.19.0.dev0。两版 ``glm5_next`` 有实质差异，本票以**本地检出
的实际代码**为唯一实现版本：
- 5.16 的 MLA 缓存写**展开后的** K/V（modeling 初版 ``past_key_values.update(
  key_states, value_states)``），本地 5.19 写**压缩 latent**（512 维 + 0 维
  k_rot），用时才 ``kv_b_proj`` 展开；本票只表达 5.19 的压缩 latent 缓存，
  不把 5.16 的展开缓存结论混入。
- 5.16 的 ``chunk_kimi_delta_attention`` decay mask 未屏蔽严格上三角（后续
  ``0b179b2df5`` 修复 NaN 梯度）；本地 5.19 已 mask 成 -inf。
- 5.16 的专家 ``one_hot(..., num_classes=E)``，本地 5.19 为 ``E+1``（多一个
  永远跳过的 no-expert 槽）；本票按本地 5.19 表达 ``[E+1,K,T]``。
- 5.16 的 layer type 旧名由 5.19 的 ``remap_legacy_layer_types`` 统一规范化。

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
    "GLM53FLASH_SCENARIOS",
    "verify_source_fingerprints",
    "load_hf_model_values",
    "layer_group_assignments",
    "effective_parameter_records",
    "effective_layer_types",
    "effective_mlp_layer_types",
    "effective_indexer_types",
    "effective_linear_attention_config",
    "kda_layer_indices",
    "dsa_layer_indices",
]

# GLM-5.3-Flash 的少量有区别场景（不是长度组合枚举）。本模型同时存在
# 「固定大小 KDA 状态」与「随历史增长的 MLA/DSA 缓存」，场景分别检验：
# 空缓存 / 非零历史 / 历史远超 index_topk（池 top-k 饱和 + tail）/ 多 token。
GLM53FLASH_SCENARIOS = (
    {
        "id": "prefill-empty",
        "label": "空缓存 Prefill（示例 seqlen=8）",
        "seqlen": 8,
        "kvcachelen": 0,
        "note": "kvcachelen=0 即空缓存起步；seqlen=8 只是示例取值，结构 shape 仍保留符号 seqlen。"
                "Lkv=8 时 DSA 候选池只有 2 个，池 top-k 全选、tail 为空。",
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
        "note": "kvcachelen=4096 是示例逻辑历史长度；MLA latent 与 indexer packed 缓存都线性保留，"
                "Lkv=Lidx=4097；KDA 层状态大小不变（不随历史增长）。",
    },
    {
        "id": "decode-long",
        "label": "长历史单 token Decode（kvcachelen=131072）",
        "seqlen": 1,
        "kvcachelen": 131072,
        "note": "历史远超 index_topk=2048：DSA 只选 512 个池（2048 token）+ 当前不完整池 tail；"
                "Lkv 仍为全部历史+新 token（latent 缓存无压缩条目）。",
    },
    {
        "id": "decode-multi",
        "label": "非零历史多 token Decode（seqlen=4，kvcachelen=4092）",
        "seqlen": 4,
        "kvcachelen": 4092,
        "note": "多 token Decode：检查统一结构按 seqlen 表达，而不是固定 +1/T=1。"
                "Lkv=4096 恰为 kpool 的整数倍，tail 为空。",
    },
)

# 官方模型仓库身份（HF API /api/models/zai-org/GLM-5.3-Flash 于 2026-10-29 返回的 sha）。
MODEL_REPOSITORY = {
    "repo_id": "zai-org/GLM-5.3-Flash",
    "revision": "46a050bec13002ad29ee795c16e61cf4fa3c4a9c",
    "last_modified": "2026-10-15T06:32:52.000Z",
    "pipeline_tag": "image-text-to-text",
    "library_name": "transformers",
    "model_type": "glm5_next",
    "architectures": ["Glm5NextForConditionalGeneration"],
    "text_model_type": "glm5_next_text",
    "checkpoint_transformers_version": "5.16.0",
    "note": (
        "配置与模型卡按上述 revision 取得；只取 config.json（69,416 B）与 README.md（8,027 B）"
        "两个小文件，不下载权重分片。视觉编码器字段存在但不在本票分析范围。"
    ),
}

# 所选 eager forward 路径实际读取的定义闭包（相对工作区根）。
SOURCE_FILES = {
    "config.json": "models/zai-org/GLM-5.3-Flash/config.json",
    "modeling_glm5_next.py": (
        "hflib/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py"
    ),
    "configuration_glm5_next.py": (
        "hflib/transformers/src/transformers/models/glm5_next/configuration_glm5_next.py"
    ),
    "cache_utils.py": "hflib/transformers/src/transformers/cache_utils.py",
    "masking_utils.py": "hflib/transformers/src/transformers/masking_utils.py",
    "activations.py": "hflib/transformers/src/transformers/activations.py",
    "configuration_utils.py": "hflib/transformers/src/transformers/configuration_utils.py",
}

# 已分析来源指纹（SHA256，本地检出 536ecc00… / 官方 config revision 见上）。
# 来源漂移（任一文件变化）时构建具名拒绝、要求重新分析后更新指纹，而不是拿旧
# 结论继续产出。
ANALYZED_SOURCE_FINGERPRINTS = {
    "config.json": "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f",
    "modeling_glm5_next.py": "caa38805e6bb7f8772b7016f98897ca0408090fd00c314ea6c8e63c543b586cf",
    "configuration_glm5_next.py": "0a4df8f0238fd6e903fc94e7d89c0feee0bbea39bd8c92ccb42506e1b135d644",
    "cache_utils.py": "2bef552bc39e8a4dc7ddfcf25740e6a4a7c4b836d92f5e56629a8ebfcd9cb246",
    "masking_utils.py": "c1d0e1e61754db819382bb03acdd2ab38c1bed3e9e9cf010918d2bf2771a8415",
    "activations.py": "5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665",
    "configuration_utils.py": "6cde90e0992d0dcf0216ab773254c5dd243e04f3c896d6e3d88a2b8be0d46567",
}


class HfConfigError(ValueError):
    """HF 根配置缺失/非法，或与已解释的身份不一致（具名拒绝）。"""


def verify_source_fingerprints(workspace_root: Path) -> None:
    """复算 SOURCE_FILES 各输入 SHA256 并与已分析指纹核对；漂移具名拒绝。"""
    import hashlib

    for key, relpath in SOURCE_FILES.items():
        path = Path(workspace_root) / relpath
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


def _load_root_config(workspace_root: Path) -> dict:
    path = Path(workspace_root) / SOURCE_FILES["config.json"]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HfConfigError(f"HF 根配置不可读: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HfConfigError(f"HF 根配置不是合法 JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise HfConfigError(f"HF 根配置顶层必须是对象: {path}")
    if raw.get("model_type") != "glm5_next":
        raise HfConfigError(
            f"HF 根配置 model_type 必须是 glm5_next（GLM-5.3-Flash 官方实现）: {path}"
        )
    if raw.get("architectures") != ["Glm5NextForConditionalGeneration"]:
        raise HfConfigError(
            f"HF 根配置 architectures 必须是 ['Glm5NextForConditionalGeneration']: {path}"
        )
    text = raw.get("text_config")
    if not isinstance(text, dict):
        raise HfConfigError("根配置缺少 text_config 对象（本票只分析语言主干）")
    if text.get("model_type") != "glm5_next_text":
        raise HfConfigError(
            f"text_config.model_type 必须是 glm5_next_text: {text.get('model_type')!r}"
        )
    vision = raw.get("vision_config")
    if not isinstance(vision, dict) or vision.get("model_type") != "glm5_next_vision":
        raise HfConfigError("根配置缺少合法的 vision_config（多模态复合配置的结构事实）")
    return raw


def effective_layer_types(text: dict) -> list[str]:
    """注意力层型：显式列表，按源码规则规范化旧名。

    ``Glm5NextTextConfig.__post_init__``：``full_attention`` → ``indexed_attention``
    （configuration:170-172）；随后 ``PreTrainedConfig.__post_init__`` 的
    ``remap_legacy_layer_types`` 把 ``deepseek_sparse_attention`` 也规范为
    ``indexed_attention``（configuration_utils.py:87-134）。本模型显式配置使用旧名，
    若不按源码重映射会在 ``causal_mask_mapping[...]`` 直接 KeyError——因此这里必须
    复刻源码规则，不能按配置字面值判断层型。
    """
    n = int(text["num_hidden_layers"])
    explicit = text.get("layer_types")
    if not isinstance(explicit, list) or len(explicit) != n:
        raise HfConfigError(f"layer_types 必须是长度 {n} 的数组（实得 {explicit!r}）")
    remap = {
        "deepseek_sparse_attention": "indexed_attention",
        "full_attention": "indexed_attention",
    }
    types: list[str] = []
    for index, value in enumerate(explicit):
        mapped = remap.get(value, value)
        if mapped not in ("linear_attention", "indexed_attention"):
            raise HfConfigError(f"layer_types[{index}]={value!r} 不在已分析层型内")
        types.append(mapped)
    if "linear_attention" not in types or "indexed_attention" not in types:
        raise HfConfigError("本模型必须是 KDA 与 MLA/DSA 的混合层型（缺一种即身份不符）")
    return types


def effective_mlp_layer_types(text: dict) -> list[str]:
    """MLP 层型：显式列表 > ``first_k_dense_replace`` 推导；不一致具名拒绝。"""
    n = int(text["num_hidden_layers"])
    explicit = text.get("mlp_layer_types")
    if explicit is not None:
        if not isinstance(explicit, list) or len(explicit) != n:
            raise HfConfigError(f"mlp_layer_types 必须是长度 {n} 的数组（实得 {explicit!r}）")
        for index, value in enumerate(explicit):
            if value not in ("dense", "sparse"):
                raise HfConfigError(f"mlp_layer_types[{index}]={value!r} 不在 'dense'/'sparse' 内")
        n_dense = min(int(text.get("first_k_dense_replace", 3)), n)
        derived = ["dense"] * n_dense + ["sparse"] * (n - n_dense)
        if list(explicit) != derived:
            raise HfConfigError(
                "根配置 mlp_layer_types 与 first_k_dense_replace 推导不一致；不静默选择一边"
            )
        return list(explicit)
    n_dense = min(int(text.get("first_k_dense_replace", 3)), n)
    return ["dense"] * n_dense + ["sparse"] * (n - n_dense)


def effective_indexer_types(text: dict) -> list[str]:
    """逐层 DSA indexer 模式；本配置必须全为 ``full``（不激活 shared 分支）。

    shared 分支在本地源码中真实存在（不实例化 indexer、消费 prev_topk_indices），
    但本票没有该配置下的真实层，不能凭空生成第四种模板。若未来配置出现 shared，
    这里具名拒绝并要求重新分析，而不是静默套用 full 层结构。
    """
    n = int(text["num_hidden_layers"])
    explicit = text.get("indexer_types")
    if not isinstance(explicit, list) or len(explicit) != n:
        raise HfConfigError(f"indexer_types 必须是长度 {n} 的数组（实得 {explicit!r}）")
    for index, value in enumerate(explicit):
        if value not in ("full", "shared"):
            raise HfConfigError(f"indexer_types[{index}]={value!r} 不在 'full'/'shared' 内")
    if any(value != "full" for value in explicit):
        raise HfConfigError(
            "本票只分析官方配置中 indexer_types 全为 full 的 DSA 层；出现 shared 需要"
            "新增 shared 模板与 prev_topk 传递分析，不能把 full 层结构套上去"
        )
    return list(explicit)


def effective_linear_attention_config(text: dict) -> dict:
    """KDA 超参（``linear_attn_config`` 覆盖配置类默认值，configuration:189-197）。"""
    raw = text.get("linear_attn_config")
    if not isinstance(raw, dict):
        raise HfConfigError("根配置缺少 linear_attn_config 对象（KDA 超参来源）")
    required = ("num_heads", "head_dim", "short_conv_kernel_size", "gate_lower_bound")
    result = {}
    for key in required:
        value = raw.get(key)
        if value is None:
            raise HfConfigError(f"linear_attn_config 缺少 {key}")
        result[key] = value
    num_heads = int(result["num_heads"])
    head_dim = int(result["head_dim"])
    kernel = int(result["short_conv_kernel_size"])
    if num_heads <= 0 or head_dim <= 0 or kernel <= 0:
        raise HfConfigError("linear_attn_config 的 heads/head_dim/kernel 必须为正")
    return {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "short_conv_kernel_size": kernel,
        "gate_lower_bound": float(result["gate_lower_bound"]),
        "kda_layers": list(raw.get("kda_layers") or []),
        "full_attn_layers": list(raw.get("full_attn_layers") or []),
    }


def kda_layer_indices(values: "HfModelValues") -> list[int]:
    return [i for i in range(values.num_hidden_layers) if values.layer_types[i] == "linear_attention"]


def dsa_layer_indices(values: "HfModelValues") -> list[int]:
    return [i for i in range(values.num_hidden_layers) if values.layer_types[i] == "indexed_attention"]


@dataclass(frozen=True)
class HfModelValues:
    """layer_idx 视角的 GLM-5.3-Flash 文本 HF 有效配置值（静态解释，非运行时探值）。"""

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
    sub_norm_eps: float
    hidden_act: str
    use_cache: bool
    max_position_embeddings: int
    tie_word_embeddings: bool
    pad_token_id: int
    dtype: str
    # MLA / DSA 注意力（NoPE）
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    attention_bias: bool
    attention_dropout: float
    attention_scaling: float
    no_rope: bool
    # DSA indexer（k-pool 压缩）
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_kpool_always_select_tail: bool
    indexer_types: list[str]
    # KDA 线性注意力
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_dim: int
    linear_lower_bound: float
    # mHC
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
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
    topk_method: str
    swiglu_limit: float
    # 层规划
    layer_types: list[str]
    mlp_layer_types: list[str]
    # 声明但明确不在本票范围
    quantization_config: dict
    num_nextn_predict_layers: int
    vision_depth: int
    image_token_id: int

    def layer_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def mlp_layer_type(self, layer_idx: int) -> str:
        return self.mlp_layer_types[layer_idx]

    def indexer_type(self, layer_idx: int) -> str:
        if self.layer_type(layer_idx) != "indexed_attention":
            return "none"
        return self.indexer_types[layer_idx]


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


def load_hf_model_values(
    workspace_root: str | Path,
    model_key: str = "glm_5_3_flash_hf",
) -> HfModelValues:
    """读取根 config.json 并静态解释为 Flash 文本有效配置；失配/缺字段具名拒绝。"""
    workspace_root = Path(workspace_root)
    raw = _load_root_config(workspace_root)
    text = raw["text_config"]
    vision = raw["vision_config"]

    def _require(name: str):
        value = text.get(name)
        if value is None:
            raise HfConfigError(f"text_config 缺少必要字段 {name!r}")
        return value

    layer_types = effective_layer_types(text)
    mlp_types = effective_mlp_layer_types(text)
    indexer_types = effective_indexer_types(text)
    linear = effective_linear_attention_config(text)

    n = int(_require("num_hidden_layers"))
    kda_layers = [i for i in range(n) if layer_types[i] == "linear_attention"]
    dsa_layers = [i for i in range(n) if layer_types[i] == "indexed_attention"]
    declared_kda = linear["kda_layers"]
    declared_dsa = linear["full_attn_layers"]
    if declared_kda and declared_kda != kda_layers:
        raise HfConfigError(
            f"linear_attn_config.kda_layers={declared_kda} 与 layer_types 推导的 "
            f"KDA 层 {kda_layers} 不一致"
        )
    if declared_dsa and declared_dsa != dsa_layers:
        raise HfConfigError(
            f"linear_attn_config.full_attn_layers={declared_dsa} 与 layer_types 推导的 "
            f"DSA 层 {dsa_layers} 不一致"
        )

    qk_nope = int(_require("qk_nope_head_dim"))
    qk_rope = int(_require("qk_rope_head_dim"))
    declared_qk = text.get("qk_head_dim")
    qk_head_dim = qk_nope + qk_rope
    if declared_qk is not None and int(declared_qk) != qk_head_dim:
        raise HfConfigError(
            f"text_config qk_head_dim={declared_qk} 与 qk_nope+qk_rope={qk_head_dim} 不一致"
        )
    if qk_rope != 0:
        raise HfConfigError(
            f"GLM-5.3-Flash 的 DSA 层要求 NoPE（qk_rope_head_dim=0），实得 {qk_rope}；"
            "若配置改为非零，必须重新分析 RoPE 与缓存形状"
        )
    config_head_dim_raw = int(_require("head_dim"))
    # __post_init__ :199-202：head_dim 被强制指向 qk_rope_head_dim（本配置 0）。
    num_heads = int(_require("num_attention_heads"))
    num_kv = int(_require("num_key_value_heads"))
    if num_heads != num_kv:
        raise HfConfigError(
            f"本地实现要求 num_attention_heads == num_key_value_heads（{num_heads} vs {num_kv}）"
        )
    index_kpool = int(_require("index_kpool"))
    index_topk = int(_require("index_topk"))
    if index_kpool < 1 or index_topk % index_kpool != 0:
        raise HfConfigError(
            f"index_topk={index_topk} 必须能被 index_kpool={index_kpool} 整除（源码 validate_architecture）"
        )
    if not bool(text.get("index_kpool_always_select_tail", True)):
        raise HfConfigError(
            "本分析未覆盖 index_kpool_always_select_tail=false 的变体：K_sel 的池+tail "
            "推导式不同，不能套用"
        )
    if not bool(text.get("mhc", True)):
        raise HfConfigError(
            "本地 5.19 实现无条件构造 attn_hc/ffn_hc（不读 mhc 开关）；配置声明 mhc=false "
            "与所选源码不一致，拒绝混用"
        )

    rms_norm_eps = float(_require("rms_norm_eps"))
    if rms_norm_eps != 1e-5:
        raise HfConfigError(
            f"本票锁定的实现事实是主 RMSNorm eps=1e-5（实得 {rms_norm_eps}）；"
            "若配置变化需重新核对 q_a/kv_a/o_norm/hc_norm 的实际 eps"
        )

    hidden_act = str(text.get("hidden_act", "silu"))
    if hidden_act != "silu":
        raise HfConfigError(f"本票只分析 hidden_act=silu 的 SwiGLU（实得 {hidden_act!r}）")

    return HfModelValues(
        model_key=model_key,
        display_name="GLM-5.3-Flash (zai-org/GLM-5.3-Flash, HF transformers glm5_next 文本主干)",
        implementation="transformers",
        source_revision=_git_head(workspace_root / "hflib" / "transformers"),
        config_path=SOURCE_FILES["config.json"],
        model_revision=MODEL_REPOSITORY["revision"],
        checkpoint_transformers_version=MODEL_REPOSITORY["checkpoint_transformers_version"],
        hidden_size=int(_require("hidden_size")),
        num_hidden_layers=n,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv,
        head_dim=qk_rope,
        config_head_dim_raw=config_head_dim_raw,
        vocab_size=int(_require("vocab_size")),
        rms_norm_eps=rms_norm_eps,
        sub_norm_eps=1e-6,
        hidden_act=hidden_act,
        use_cache=bool(text.get("use_cache", True)),
        max_position_embeddings=int(_require("max_position_embeddings")),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", text.get("tie_word_embeddings", False))),
        pad_token_id=int(text.get("pad_token_id", 154820)),
        dtype=str(text.get("dtype", "bfloat16")),
        q_lora_rank=int(_require("q_lora_rank")),
        kv_lora_rank=int(_require("kv_lora_rank")),
        qk_head_dim=qk_head_dim,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=int(_require("v_head_dim")),
        attention_bias=bool(text.get("attention_bias", False)),
        attention_dropout=float(text.get("attention_dropout", 0.0)),
        attention_scaling=qk_head_dim ** -0.5,
        no_rope=qk_rope == 0,
        index_n_heads=int(_require("index_n_heads")),
        index_head_dim=int(_require("index_head_dim")),
        index_topk=index_topk,
        index_kpool=index_kpool,
        index_kpool_always_select_tail=bool(text.get("index_kpool_always_select_tail", True)),
        indexer_types=indexer_types,
        linear_num_heads=linear["num_heads"],
        linear_head_dim=linear["head_dim"],
        linear_conv_kernel_dim=linear["short_conv_kernel_size"],
        linear_lower_bound=linear["gate_lower_bound"],
        hc_mult=int(_require("hc_mult")),
        hc_eps=float(_require("hc_eps")),
        hc_sinkhorn_iters=int(_require("hc_sinkhorn_iters")),
        first_k_dense_replace=int(text.get("first_k_dense_replace", 3)),
        intermediate_size=int(_require("intermediate_size")),
        moe_intermediate_size=int(_require("moe_intermediate_size")),
        n_shared_experts=int(text.get("n_shared_experts", 1)),
        n_routed_experts=int(text.get("n_routed_experts", 288)),
        num_experts_per_tok=int(_require("num_experts_per_tok")),
        scoring_func=str(text.get("scoring_func", "sigmoid")),
        norm_topk_prob=bool(text.get("norm_topk_prob", True)),
        routed_scaling_factor=float(_require("routed_scaling_factor")),
        n_group=int(text.get("n_group", 1)),
        topk_group=int(text.get("topk_group", 1)),
        moe_router_dtype=str(text.get("moe_router_dtype", "float32")),
        topk_method=str(text.get("topk_method", "noaux_tc")),
        swiglu_limit=float(_require("swiglu_limit")),
        layer_types=layer_types,
        mlp_layer_types=mlp_types,
        quantization_config=raw.get("quantization_config", {}),
        num_nextn_predict_layers=int(text.get("num_nextn_predict_layers", 0)),
        vision_depth=int(vision.get("depth", 0)),
        image_token_id=int(raw.get("image_token_id", 0)),
    )


def effective_parameter_records(values: HfModelValues) -> list[dict]:
    """逐参数 raw 字段/默认/派生规则与来源（供参数符号区联查）。"""
    modeling = SOURCE_FILES["modeling_glm5_next.py"]
    configuration = SOURCE_FILES["configuration_glm5_next.py"]
    config_json = SOURCE_FILES["config.json"]
    cache_utils = SOURCE_FILES["cache_utils.py"]
    masking = SOURCE_FILES["masking_utils.py"]
    config_utils = SOURCE_FILES["configuration_utils.py"]
    activations = SOURCE_FILES["activations.py"]

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
                "model_type": "glm5_next",
                "architectures": ["Glm5NextForConditionalGeneration"],
                "text_config.model_type": "glm5_next_text",
                "pipeline_tag": MODEL_REPOSITORY["pipeline_tag"],
            },
            "rule": "官方模型仓库 revision 与根配置 model_type/architectures 双条件锁定；"
            "不以目录名或相似模型（GLM-5.3/GLM-5）代替",
            "source_refs": [
                cfg("model_type", "根配置字段"),
                cfg("architectures", "根配置字段"),
                cfg("text_config.model_type", "文本子配置字段"),
            ],
        },
        {
            "field": "text_vision_boundary",
            "value": {
                "text_config": "glm5_next_text",
                "vision_config": "glm5_next_vision",
                "vision_depth": values.vision_depth,
                "image_token_id": values.image_token_id,
            },
            "rule": "多模态复合配置：本票只分析 text_config（语言主干），vision_config 与"
            "图像/视频 placeholder 替换路径不在范围",
            "source_refs": [
                cfg("text_config", "文本子配置"),
                cfg("vision_config", "视觉子配置（仅记录边界，不分析）"),
                cfg("image_token_id", "图像占位 token（不在本主干）"),
            ],
        },
        {
            "field": "hidden_size",
            "value": values.hidden_size,
            "rule": "text_config 直接字段",
            "source_refs": [cfg("text_config.hidden_size", "文本子配置字段"),
                            rule("101", "配置类字段声明")],
        },
        {
            "field": "num_hidden_layers",
            "value": values.num_hidden_layers,
            "rule": "text_config 直接字段；主干层数 45（0..44）。checkpoint 的 layer 45 是 "
            "MTP，被显式忽略（modeling:1378），forward 只循环 0..44",
            "source_refs": [
                cfg("text_config.num_hidden_layers", "文本子配置字段"),
                rule("105", "配置类字段声明"),
                src(modeling, "1378", "_keys_to_ignore_on_load_unexpected = [r\"layers\\.45\\.\", ...]"),
                src(modeling, "1495-1496", "for i, decoder_layer in enumerate(self.layers[:num_hidden_layers])"),
            ],
        },
        {
            "field": "layer_types",
            "value": values.layer_types,
            "rule": "显式 45 项；34 个 linear_attention（KDA）+ 11 个旧名 "
            "deepseek_sparse_attention。源码两处重映射（full_attention / deepseek_sparse_attention "
            "→ indexed_attention）后才与 cache/mask 分派键一致；不能按配置字面值判断",
            "source_refs": [
                cfg("text_config.layer_types", "文本子配置显式数组（旧名）"),
                rule("164-172", "__post_init__：full_attention → indexed_attention"),
                src(config_utils, "87-134", "remap_legacy_layer_types：deepseek_sparse_attention → indexed_attention"),
                src(modeling, "1488-1491", "causal_mask_mapping 只含 indexed_attention/linear_attention 两个键"),
            ],
        },
        {
            "field": "linear_attn_config",
            "value": {
                "num_heads": values.linear_num_heads,
                "head_dim": values.linear_head_dim,
                "short_conv_kernel_size": values.linear_conv_kernel_dim,
                "gate_lower_bound": values.linear_lower_bound,
            },
            "rule": "text_config.linear_attn_config 覆盖配置类默认值（configuration:189-197）；"
            "KDA 的 q/k/v 宽度 = 64×128 = 8192，conv 通道 = 3×8192",
            "source_refs": [
                cfg("text_config.linear_attn_config", "文本子配置"),
                rule("189-197", "__post_init__ 读取 head_dim/num_heads/kernel/gate_lower_bound"),
                src(modeling, "620-659", "Glm5NextTextLinearAttention 形状"),
            ],
        },
        {
            "field": "mlp_layer_types",
            "value": values.mlp_layer_types,
            "rule": "显式 45 项：前 3 层 dense，其余 sparse（= first_k_dense_replace=3）",
            "source_refs": [
                cfg("text_config.mlp_layer_types", "文本子配置显式数组"),
                cfg("text_config.first_k_dense_replace", "文本子配置字段"),
                rule("159-162", "__post_init__ mlp_layer_types 派生"),
                src(modeling, "1289-1291", "sparse → Glm5NextTextMoE，否则 Glm5NextTextMLP"),
            ],
        },
        {
            "field": "num_attention_heads",
            "value": values.num_attention_heads,
            "rule": "text_config 直接字段；DSA 的 64 个 query 头，KV 头同为 64"
            "（num_key_value_groups=1，repeat_kv 恒等）",
            "source_refs": [
                cfg("text_config.num_attention_heads", "文本子配置字段"),
                src(modeling, "1123", "self.num_key_value_groups = num_attention_heads // num_key_value_heads"),
                src(modeling, "1063-1072", "repeat_kv：n_rep==1 直接返回"),
            ],
        },
        {
            "field": "num_key_value_heads",
            "value": values.num_key_value_heads,
            "rule": "text_config 直接字段；与 query 头相同，K/V 展开为 64 头时不做 KV 复制",
            "source_refs": [cfg("text_config.num_key_value_heads", "文本子配置字段")],
        },
        {
            "field": "qk_head_size",
            "value": {"qk_head_dim": values.qk_head_dim,
                      "qk_nope_head_dim": values.qk_nope_head_dim,
                      "qk_rope_head_dim": values.qk_rope_head_dim,
                      "head_dim": values.head_dim,
                      "head_dim_raw": values.config_head_dim_raw},
            "rule": "qk_head_dim = nope 256 + rope 0 = 256；__post_init__ 把运行期 head_dim "
            "指向 qk_rope_head_dim=0，因此 NoPE；根配置 head_dim=0 不参与 RoPE",
            "source_refs": [
                cfg("text_config.qk_nope_head_dim", "文本子配置字段"),
                cfg("text_config.qk_rope_head_dim", "文本子配置字段（0）"),
                rule("199-202", "kwargs.pop('head_dim'); self.head_dim = self.qk_rope_head_dim"),
                rule("206-226", "validate_architecture：qk_rope_head_dim > 0 直接拒绝"),
                src(modeling, "1186-1192", "q 只有 256 维；kv_a 输出 512+0；无 RoPE 调用"),
            ],
        },
        {
            "field": "v_head_dim",
            "value": values.v_head_dim,
            "rule": "text_config 直接字段；kv_b_proj 输出 Nh×(256+256)，随后拆 k_nope/value",
            "source_refs": [
                cfg("text_config.v_head_dim", "文本子配置字段"),
                src(modeling, "1136-1140", "kv_b_proj 输出 num_heads*(qk_nope+v_head)"),
                src(modeling, "1164-1166", "split([qk_nope_head_dim, v_head_dim])"),
            ],
        },
        {
            "field": "q_lora_rank",
            "value": values.q_lora_rank,
            "rule": "text_config 直接字段；q_a_proj → q_a_layernorm（RMSNorm，eps=rms_norm_eps）→ q_b_proj",
            "source_refs": [
                cfg("text_config.q_lora_rank", "文本子配置字段"),
                src(modeling, "1126-1128", "q_a_proj/q_a_layernorm/q_b_proj"),
            ],
        },
        {
            "field": "kv_lora_rank",
            "value": values.kv_lora_rank,
            "rule": "text_config 直接字段；缓存写的是 kv_a_layernorm 后的压缩 latent（512 维）；"
            "values 槽的 k_rot 为 0 维（NoPE）",
            "source_refs": [
                cfg("text_config.kv_lora_rank", "文本子配置字段"),
                src(modeling, "1130-1135", "kv_a_proj_with_mqa(kv_lora_rank + qk_rope_head_dim=0)"),
                src(modeling, "1194-1198", "缓存在展开前写入压缩 latent"),
            ],
        },
        {
            "field": "vocab_size",
            "value": values.vocab_size,
            "rule": "text_config 直接字段；embedding 与 lm_head 的词表规模",
            "source_refs": [cfg("text_config.vocab_size", "文本子配置字段"),
                            src(modeling, "2145", "lm_head = Linear(hidden_size, vocab_size, bias=False)")],
        },
        {
            "field": "rms_norm_eps",
            "value": values.rms_norm_eps,
            "rule": "text_config 直接字段；input/post_attention/final RMSNorm、q_a_layernorm、"
            "kv_a_layernorm、o_norm、mHC input_norm 都用它（1e-5）。indexer k_norm 是 "
            "LayerNorm(eps=1e-6)（源码字面量），不是同一个 eps",
            "source_refs": [
                cfg("text_config.rms_norm_eps", "文本子配置字段"),
                src(modeling, "1127", "q_a_layernorm = RMSNorm(..., eps=config.rms_norm_eps)"),
                src(modeling, "1135", "kv_a_layernorm = RMSNorm(..., eps=config.rms_norm_eps)"),
                src(modeling, "1293-1294", "input/post_attention_layernorm 用 config.rms_norm_eps"),
                src(modeling, "799", "indexer k_norm = LayerNorm(head_dim, eps=1e-6)"),
            ],
        },
        {
            "field": "hidden_act",
            "value": values.hidden_act,
            "rule": "text_config 直接字段；SiLU（稠密 MLP、专家、共享专家）",
            "source_refs": [cfg("text_config.hidden_act", "文本子配置字段"),
                            src(activations, "ACT2FN['silu']", "SiLUActivation")],
        },
        {
            "field": "swiglu_limit",
            "value": values.swiglu_limit,
            "rule": "text_config 直接字段；gate clamp(max=10)、up clamp(±10) 后才 SiLU(gate)*up"
            "（稠密 MLP / 专家 / 共享专家 / 视觉 MLP 同一规则）",
            "source_refs": [
                cfg("text_config.swiglu_limit", "文本子配置字段"),
                src(modeling, "95-103", "Glm5NextTextMLP.forward：gate/up clamp"),
                src(modeling, "136-141", "Glm5NextTextExperts._apply_gate：gate/up clamp"),
                src(modeling, "195-206", "共享专家复用 Glm5NextTextMLP"),
            ],
        },
        {
            "field": "hc_mult",
            "value": values.hc_mult,
            "rule": "text_config 直接字段；每个 decoder layer 的 attn/ffn 两处 mHC 都把 "
            "4 路流坍缩成单流，再把子层输出展开/混合回 4 路（不是普通残差）",
            "source_refs": [
                cfg("text_config.hc_mult", "文本子配置字段"),
                src(modeling, "271-329", "Glm5NextTextHyperConnection"),
                src(modeling, "1296-1297", "self.attn_hc / self.ffn_hc"),
                src(modeling, "1312-1346", "两处 post/comb 混合"),
            ],
        },
        {
            "field": "hc_sinkhorn",
            "value": {"hc_sinkhorn_iters": values.hc_sinkhorn_iters, "hc_eps": values.hc_eps},
            "rule": "comb 先 softmax + eps，再做 20 轮 Sinkhorn 行列归一，约束为双随机矩阵",
            "source_refs": [
                cfg("text_config.hc_sinkhorn_iters", "文本子配置字段"),
                cfg("text_config.hc_eps", "文本子配置字段"),
                src(modeling, "313-324", "pre/post/comb 与 Sinkhorn 迭代"),
            ],
        },
        {
            "field": "hc_head",
            "value": "mean over hc_mult streams",
            "rule": "网络末端把 4 路流按均值收拢，再做 Final RMSNorm；"
            "不是 DeepSeek 的加权 HyperHead",
            "source_refs": [
                src(modeling, "332-336", "Glm5NextTextHyperHead.forward: hidden_streams.mean(dim=2)"),
                src(modeling, "1509", "hidden_states = self.norm(self.hc_head(hidden_states))"),
            ],
        },
        {
            "field": "index_n_heads",
            "value": values.index_n_heads,
            "rule": "text_config 直接字段；DSA indexer 的 32 个打分头",
            "source_refs": [cfg("text_config.index_n_heads", "文本子配置字段"),
                            src(modeling, "797-801", "wq_b/wk/k_norm/weights_proj 形状")],
        },
        {
            "field": "index_head_dim",
            "value": values.index_head_dim,
            "rule": "text_config 直接字段；indexer 头维 128；k_norm 是带 weight/bias 的 LayerNorm",
            "source_refs": [cfg("text_config.index_head_dim", "文本子配置字段"),
                            src(modeling, "799", "self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)")],
        },
        {
            "field": "index_topk",
            "value": values.index_topk,
            "rule": "text_config 直接字段；每 query 的预算 2048 个原始 token。源码按 kpool 压缩池"
            "选择：select_k = min(index_topk//kpool, 池数)，展开成原始 token 后再追 tail",
            "source_refs": [
                cfg("text_config.index_topk", "文本子配置字段"),
                src(modeling, "883", "select_k = min(self.index_topk // self.index_kpool, ...)"),
                src(modeling, "894-911", "池展开 + tail + pad 到 index_topk + kpool - 1"),
            ],
        },
        {
            "field": "index_kpool",
            "value": values.index_kpool,
            "rule": "text_config 直接字段 4；候选按 4 个连续 token 压成一个池（softmax 加权键），"
            "池内 token 不可单独选择；不完整尾池作为 tail 直接追加",
            "source_refs": [
                cfg("text_config.index_kpool", "文本子配置字段"),
                cfg("text_config.index_kpool_always_select_tail", "文本子配置字段（true）"),
                src(modeling, "935-1008", "get_pooled_states：每 kpool 个 token 一池"),
                src(modeling, "1010-1060", "append_visible_tail：当前不完整池作为 tail"),
            ],
        },
        {
            "field": "indexer_types",
            "value": values.indexer_types,
            "rule": "显式 45 项且全为 full：11 个 DSA 层各自实例化 indexer；"
            "shared 分支（复用上一层 top-k）在本配置没有真实层，未分析",
            "source_refs": [
                cfg("text_config.indexer_types", "文本子配置显式数组"),
                src(modeling, "1149-1153", "skip_topk/indexer/next_skip_topk"),
                src(modeling, "1200-1210", "full 层跑 indexer；shared 层消费 prev_topk_indices"),
            ],
        },
        {
            "field": "attention_scaling",
            "value": values.attention_scaling,
            "rule": "qk_head_dim**-0.5 = 256^-0.5 = 1/16；NoPE 没有 YaRN/mscale 修正",
            "source_refs": [
                src(modeling, "1147", "self.scaling = self.qk_head_dim ** (-0.5)"),
                src(modeling, "1088", "attn_weights = matmul(q, k^T) * scaling"),
            ],
        },
        {
            "field": "indexer_scaling",
            "value": values.index_head_dim ** -0.5,
            "rule": "indexer 打分缩放 = index_head_dim^-0.5 = 128^-0.5；再乘每头权重 n_heads^-0.5",
            "source_refs": [
                src(modeling, "801", "self.softmax_scale = self.head_dim**-0.5"),
                src(modeling, "861-866", "scores*scale；weights*n_heads^-0.5；加权求和"),
            ],
        },
        {
            "field": "intermediate_size",
            "value": values.intermediate_size,
            "rule": "text_config 直接字段；dense MLP 中间维 12288",
            "source_refs": [cfg("text_config.intermediate_size", "文本子配置字段"),
                            src(modeling, "85-103", "Glm5NextTextMLP 用 config.intermediate_size")],
        },
        {
            "field": "moe_intermediate_size",
            "value": values.moe_intermediate_size,
            "rule": "text_config 直接字段；单专家中间维 2048；共享专家宽度 = 2048×1",
            "source_refs": [
                cfg("text_config.moe_intermediate_size", "文本子配置字段"),
                src(modeling, "115-117", "gate_up_proj [E,2Es,H]；down_proj [E,H,Es]"),
                src(modeling, "195-197", "shared_experts = MLP(moe_intermediate_size * n_shared_experts)"),
            ],
        },
        {
            "field": "n_routed_experts",
            "value": values.n_routed_experts,
            "rule": "text_config 字段 n_routed_experts=288；标准名 num_local_experts 经 "
            "attribute_map 映射（configuration:98）",
            "source_refs": [
                cfg("text_config.n_routed_experts", "文本子配置字段"),
                rule("98", "attribute_map num_local_experts → n_routed_experts"),
                src(modeling, "112", "self.num_experts = config.num_local_experts"),
            ],
        },
        {
            "field": "n_shared_experts",
            "value": values.n_shared_experts,
            "rule": "text_config 直接字段；1 个共享专家（宽度 2048），与 routed 并行",
            "source_refs": [cfg("text_config.n_shared_experts", "文本子配置字段"),
                            src(modeling, "195-197", "self.shared_experts"),
                            src(modeling, "199-206", "routed + shared_experts(residuals)")],
        },
        {
            "field": "num_experts_per_tok",
            "value": values.num_experts_per_tok,
            "rule": "text_config 直接字段；每 token 选 8 个专家（专家总数 288）",
            "source_refs": [cfg("text_config.num_experts_per_tok", "文本子配置字段"),
                            src(modeling, "176", "topk(scores_for_choice, k=self.top_k)")],
        },
        {
            "field": "scoring_func",
            "value": values.scoring_func,
            "rule": "text_config 直接字段；sigmoid 打分（不是 softmax / sqrt-softplus）",
            "source_refs": [cfg("text_config.scoring_func", "文本子配置字段"),
                            src(modeling, "160", "scores = router_logits.sigmoid()")],
        },
        {
            "field": "norm_topk_prob",
            "value": values.norm_topk_prob,
            "rule": "text_config 直接字段；TopK 权重按和归一（+1e-20）",
            "source_refs": [cfg("text_config.norm_topk_prob", "文本子配置字段"),
                            src(modeling, "178-180", "topk_weights /= sum + 1e-20")],
        },
        {
            "field": "routed_scaling_factor",
            "value": values.routed_scaling_factor,
            "rule": "text_config 直接字段；归一后乘 2.5",
            "source_refs": [cfg("text_config.routed_scaling_factor", "文本子配置字段"),
                            src(modeling, "181", "topk_weights = topk_weights * routed_scaling_factor")],
        },
        {
            "field": "n_group",
            "value": values.n_group,
            "rule": "text_config 直接字段；n_group=1/topk_group=1 时源码仍执行组选择分支，"
            "单组掩码全通过（如实表达，不省略也不夸大）",
            "source_refs": [cfg("text_config.n_group", "文本子配置字段"),
                            cfg("text_config.topk_group", "文本子配置字段"),
                            src(modeling, "162-175", "group_scores → group_idx → score_mask → masked_fill")],
        },
        {
            "field": "topk_method",
            "value": values.topk_method,
            "rule": "配置声明 noaux_tc，但本地 5.19 实现不读该字段；路由固定走 "
            "sigmoid+bias+组选择+topk 的同一段代码（结构事实以源码为准）",
            "source_refs": [cfg("text_config.topk_method", "文本子配置声明值"),
                            src(modeling, "157-182", "TopkRouter.forward 不读 topk_method")],
        },
        {
            "field": "moe_router_dtype",
            "value": values.moe_router_dtype,
            "rule": "配置声明 float32；本地实现固定把 hidden/weight 转 float32 计算，不读该字段",
            "source_refs": [cfg("text_config.moe_router_dtype", "文本子配置声明值"),
                            src(modeling, "158-159", "hidden_states.type(float32) / weight.type(float32)")],
        },
        {
            "field": "use_cache",
            "value": values.use_cache,
            "rule": "text_config 直接字段；本分析启用缓存",
            "source_refs": [cfg("text_config.use_cache", "文本子配置字段"),
                            src(modeling, "1460-1461", "use_cache 且未传入时新建 DynamicCache")],
        },
        {
            "field": "cache_dispatch",
            "value": {"linear_attention": "LinearAttentionLayer",
                      "indexed_attention": "DynamicIndexedLayer"},
            "rule": "KDA 层缓存 conv/recurrent 状态（固定大小）；DSA 层缓存压缩 latent + "
            "indexer packed 状态（随历史增长）",
            "source_refs": [
                src(cache_utils, "1285-1294", "CACHE_LAYER_CLASSES 映射"),
                src(cache_utils, "1067-1149", "LinearAttentionLayer conv/recurrent states"),
                src(cache_utils, "331-380", "DynamicIndexedLayer 主 KV + indexer_keys"),
            ],
        },
        {
            "field": "mask_creation",
            "value": {"function": "create_recurrent_attention_mask",
                      "shape": "[B, seqlen] padding mask（None → 源码补 ones）"},
            "rule": "网络入口构建 2D padding mask，不是 4D 因果 mask；KDA 的因果性来自短卷积+"
            "递推顺序，DSA 的因果性来自 indexer 的 get_visible_tokens + 池选择",
            "source_refs": [
                src(modeling, "1471-1486", "create_recurrent_attention_mask；None 时补 ones"),
                src(masking, "1434-1470", "create_recurrent_attention_mask"),
                src(modeling, "915-933", "get_visible_tokens：causal & valid_keys"),
            ],
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
            "rule": "训练期 MTP 头数 1；本地 CausalLM 忽略 model.layers.45.* 与 shared_head.*，"
            "forward 不执行 MTP —— 不在本主干",
            "source_refs": [
                cfg("text_config.num_nextn_predict_layers", "文本子配置字段"),
                src(modeling, "1378", "_keys_to_ignore_on_load_unexpected"),
                src(modeling, "1495", "layers[:num_hidden_layers] 循环"),
            ],
        },
        {
            "field": "tie_word_embeddings",
            "value": values.tie_word_embeddings,
            "rule": "根配置 false；embed_tokens 与 lm_head 是两份独立权重"
            "（_tied_weights_keys 只是可绑定声明，配置关闭即不绑定）",
            "source_refs": [cfg("tie_word_embeddings", "根配置字段"),
                            src(modeling, "2138", "_tied_weights_keys 声明"),
                            src(modeling, "2145", "lm_head 独立 Linear")],
        },
    ]


# ---------------------------------------------------------------------------
# 三组真实层组（注意力层型 × MLP 类型）：归属、代表层与覆盖校验的单一事实来源
# ---------------------------------------------------------------------------

# 本配置只存在三种真实组合：KDA+dense（0/1/2）、DSA+MoE（3/7/…/43）、
# KDA+MoE（其余 31 层）。不虚构 "KDA+shared indexer" 或 "DSA+dense" 组。
LAYER_GROUP_SPECS = (
    {
        "id": "kda-dense",
        "label": "KDA 线性注意力 + Dense MLP",
        "attention_type": "linear_attention",
        "moe_type": "dense",
        "state_type": "kda",
        "indexer_type": "none",
        "representative_layer_index": 0,
    },
    {
        "id": "dsa-moe",
        "label": "MLA/DSA 稀疏注意力 + TopK MoE",
        "attention_type": "indexed_attention",
        "moe_type": "sparse",
        "state_type": "dsa",
        "indexer_type": "full",
        "representative_layer_index": 3,
    },
    {
        "id": "kda-moe",
        "label": "KDA 线性注意力 + TopK MoE",
        "attention_type": "linear_attention",
        "moe_type": "sparse",
        "state_type": "kda",
        "indexer_type": "none",
        "representative_layer_index": 4,
    },
)


def layer_group_assignments(values: HfModelValues) -> list[dict]:
    """按真实 layer_types/mlp_layer_types 推导三组层归属（保持 SPECS 顺序）。

    返回 [{id,label,attention_type,moe_type,state_type,indexer_type,
    representative_layer_index,layer_indices,count}]。任何未归属组合、空组、
    代表层不在组内、或三组未恰好覆盖全部主干层，均具名拒绝（ValueError）。
    """
    by_type: dict[tuple[str, str], list[int]] = {}
    for index in range(values.num_hidden_layers):
        by_type.setdefault(
            (values.layer_type(index), values.mlp_layer_type(index)), []
        ).append(index)
    groups: list[dict] = []
    covered: list[int] = []
    for spec in LAYER_GROUP_SPECS:
        layer_indices = by_type.pop((spec["attention_type"], spec["moe_type"]), [])
        if not layer_indices:
            raise ValueError(
                f"层组 {spec['id']} 在配置中没有真实层"
                f"（{spec['attention_type']} + {spec['moe_type']}）"
            )
        representative = spec["representative_layer_index"]
        if representative not in layer_indices:
            raise ValueError(
                f"层组 {spec['id']} 代表层 {representative} 不属于真实层集合 {layer_indices}"
            )
        if spec["attention_type"] == "indexed_attention":
            if any(values.indexer_type(i) != "full" for i in layer_indices):
                raise ValueError(
                    f"层组 {spec['id']} 含非 full indexer 层；本票只分析全 full 配置"
                )
        groups.append({
            **spec,
            "layer_indices": layer_indices,
            "count": len(layer_indices),
        })
        covered.extend(layer_indices)
    if by_type:
        leftover = {f"{attn}+{mlp}": idx for (attn, mlp), idx in by_type.items()}
        raise ValueError(f"存在未归属到任何已知层组的层型组合: {leftover}")
    expected = list(range(values.num_hidden_layers))
    if sorted(covered) != expected or len(covered) != len(set(covered)):
        raise ValueError(
            f"三组层集合未恰好覆盖 {values.num_hidden_layers} 层: 实得 {len(covered)} 项"
            f"（去重 {len(set(covered))}）"
        )
    return groups
