"""ONNX 模块图的构建原语与源码常量（不执行 HF、不加载权重）。

只包含：图构建小工具、config 派生的 RoPE 常数、逐算子标准 schema 直译 helper。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

OPSET = 18
IR_VERSION = 10          # onnxruntime 1.20.1 支持到 IR 10；onnx 1.18 默认 IR 11，必须显式降
DTYPE = "float32"        # 阅读用 FP32 特化图（见 report/state 的标注，不等于 HF 数值等价）
FLOAT = "float32"
INT64 = "int64"


# ---------------------------------------------------------------------------
# config 派生常数（静态读 config.json，不导入 HF）
# ---------------------------------------------------------------------------
def derive_compress_rope(config: dict) -> dict:
    """compress 层（CSA/HCA）的 YaRN RoPE 常数；公式取自 modeling_rope_utils。

    源码事实：configuration_deepseek_v4.__post_init__ 把 legacy rope_scaling(**yarn)
    合入 compress rope_parameters，并强制 attention_factor=1.0；modeling:774 对
    非 sliding 层使用 rope_layer_type='compress'。
    """
    rs = config.get("rope_scaling") or {}
    head_dim = config["head_dim"]
    partial = (config.get("qk_rope_head_dim") or 64) / head_dim
    dim = int(head_dim * partial)
    base = float(config.get("compress_rope_theta", 160000.0))
    factor = float(rs.get("factor", 16))
    beta_fast = float(rs.get("beta_fast", 32))
    beta_slow = float(rs.get("beta_slow", 1))
    ompe = float(rs.get("original_max_position_embeddings", 65536))
    attention_factor = 1.0  # V4 参考不乘 YaRN mscale

    def correction_dim(num_rot: float) -> float:
        return (dim * math.log(ompe / (num_rot * 2 * math.pi))) / (2 * math.log(base))

    low = math.floor(correction_dim(beta_fast))
    high = math.ceil(correction_dim(beta_slow))
    low, high = max(low, 0), min(high, dim - 1)
    idx = np.arange(dim // 2, dtype=np.float64)
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    inv_extrap = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)
    if low == high:
        high = low + 0.001
    ramp = np.clip((idx - low) / (high - low), 0.0, 1.0)
    extrap_factor = 1.0 - ramp
    inv_freq = inv_interp * (1 - extrap_factor) + inv_extrap * extrap_factor
    return {"rope_type": "yarn", "theta": base, "dim": dim, "factor": factor,
            "beta_fast": beta_fast, "beta_slow": beta_slow,
            "original_max_position_embeddings": ompe,
            "attention_factor": attention_factor,
            "inv_freq": inv_freq.astype(np.float32), "partial_rotary_factor": partial}


def derive_main_rope(config: dict) -> dict:
    head_dim = config["head_dim"]
    partial = (config.get("qk_rope_head_dim") or 64) / head_dim
    dim = int(head_dim * partial)
    base = float(config.get("rope_theta", 10000.0))
    inv_freq = (1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))).astype(np.float32)
    return {"rope_type": "default", "theta": base, "dim": dim,
            "attention_factor": 1.0, "inv_freq": inv_freq,
            "partial_rotary_factor": partial}


# ---------------------------------------------------------------------------
# 图构建器
# ---------------------------------------------------------------------------
@dataclass
class GB:
    module_id: str
    phase: str
    nodes: list = field(default_factory=list)
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    inits: list = field(default_factory=list)
    groups: dict = field(default_factory=dict)   # group -> {"label","nodes":[...]}
    _k: int = 0

    # -- 名称/节点 --
    def _uid(self) -> int:
        self._k += 1
        return self._k

    def node(self, op_type: str, inputs, outputs=None, *, group: str,
             name: str | None = None, **attrs):
        # opset 18：ReduceSum/ReduceMean/ReduceMax/Unsqueeze/Squeeze 的 axes 是输入而非属性
        if op_type in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin",
                       "Unsqueeze", "Squeeze") and "axes" in attrs:
            axes = attrs.pop("axes")
            label = outputs if isinstance(outputs, str) else (outputs or [str(self._k)])[0]
            inputs = list(inputs) + [self.const(f"{label}/axes", np.array(axes, np.int64))]
        k = self._uid()
        if outputs is None:
            outputs = [f"{group}/Out_{op_type}_{k}"]
        elif isinstance(outputs, str):
            outputs = [outputs]
        rec = {"op_type": op_type, "inputs": list(inputs),
               "outputs": list(outputs), "attrs": attrs,
               "name": name or f"{group}/{op_type}_{k}"}
        self.nodes.append(rec)
        self.groups.setdefault(group, {"label": group, "nodes": []})["nodes"].append(rec)
        return outputs[0] if len(outputs) == 1 else outputs

    def ensure_unique_names(self) -> None:
        """NodeProto.name 全图唯一（重映射/显式 output 之后也必须成立）。"""
        seen: dict[str, int] = {}
        for rec in self.nodes:
            base = rec["name"]
            n = seen.get(base, 0)
            seen[base] = n + 1
            if n:
                rec["name"] = f"{base}__{n}"

    def set_group_label(self, group: str, label: str) -> None:
        self.groups.setdefault(group, {"label": label, "nodes": []})["label"] = label

    # -- IO --
    def inp(self, name: str, shape, dtype: str = DTYPE, role: str = "activation") -> str:
        self.inputs.append({"name": name, "shape": list(shape), "dtype": dtype, "role": role})
        return name

    def out(self, name: str, shape, dtype: str = DTYPE, role: str = "activation",
            json_id: str | None = None) -> str:
        self.outputs.append({"name": name, "shape": list(shape), "dtype": dtype,
                             "role": role, "json_id": json_id or name})
        return name

    def const(self, name: str, value) -> str:
        arr = np.asarray(value)
        self.inits.append({"name": name, "array": arr})
        return name


# ---------------------------------------------------------------------------
# 标准 schema helper（op_type/属性均按 opset 18 实际算子名）
# ---------------------------------------------------------------------------
def linear(g: GB, group: str, x: str, w: str, out: str) -> str:
    """nn.Linear 无 bias：y = x @ W^T（W 为 [out,in] 外部输入）。"""
    wt = g.node("Transpose", [w], [f"{out}/W_T"], group=group, perm=[1, 0])
    return g.node("MatMul", [x, wt], [out], group=group)


def rmsnorm(g: GB, group: str, x: str, out: str, eps: float, weight: str | None = None) -> str:
    """DeepseekV4RMSNorm/UnweightedRMSNorm：x * rsqrt(mean(x^2)+eps) [* weight]（FP32）。"""
    sq = g.node("Mul", [x, x], [f"{group}/sq"], group=group)
    mean = g.node("ReduceMean", [sq], [f"{group}/mean"], group=group, axes=[-1], keepdims=1)
    eps_c = g.const(f"{group}/eps", np.float32(eps))
    var = g.node("Add", [mean, eps_c], [f"{group}/var_eps"], group=group)
    std = g.node("Sqrt", [var], [f"{group}/std"], group=group)
    y = g.node("Div", [x, std], [f"{group}/normed"], group=group)
    if weight:
        y = g.node("Mul", [y, weight], [out], group=group)
        return y
    return g.node("Identity", [y], [out], group=group)


def rope_cos_sin(g: GB, group: str, pos: str, rope: dict, out_cos: str, out_sin: str) -> None:
    """cos/sin：position_ids[1,T] × inv_freq[Rk/2] → [1,T,Rk/2]（inv_freq 是派生常数）。"""
    inv = g.const(f"{group}/inv_freq", rope["inv_freq"])
    posf = g.node("Cast", [pos], [f"{group}/pos_f32"], group=group, to=1)  # 1=float
    pos3 = g.node("Unsqueeze", [posf], [f"{group}/pos_u"], group=group, axes=[-1])
    freqs = g.node("Mul", [pos3, inv], [f"{group}/freqs"], group=group)
    cos = g.node("Cos", [freqs], [out_cos], group=group)
    sin = g.node("Sin", [freqs], [out_sin], group=group)
    if rope["attention_factor"] != 1.0:
        k = g.const(f"{group}/attn_scale", np.float32(rope["attention_factor"]))
        g.node("Mul", [cos, k], [out_cos], group=group)
        g.node("Mul", [sin, k], [out_sin], group=group)


def rope_apply(g: GB, group: str, x: str, cos: str, sin: str, rk: int, dh: int,
               out: str, sign: float = 1.0) -> str:
    """interleaved partial RoPE：nope | rotate_half(rope)*sin + rope*cos（HF :339-356）。

    sign=-1 时为逆旋转（attn_out 用 position=-i）。
    """
    # cos/sin [1,T,Rk/2] → [1,T,Rk] → [1,1,T,Rk]
    c4 = g.node("Unsqueeze", [cos], [f"{group}/cos_u"], group=group, axes=[-1])
    c4 = g.node("Tile", [c4, g.const(f"{group}/tile2", np.array([1, 1, 1, 2], np.int64))],
                [f"{group}/cos_tile"], group=group)
    c4 = g.node("Reshape", [c4, g.const(f"{group}/cos_shape", np.array([0, 0, rk], np.int64))],
                [f"{group}/cos_flat"], group=group)
    c4 = g.node("Unsqueeze", [c4], [f"{group}/cos_b"], group=group, axes=[1])
    s4 = g.node("Unsqueeze", [sin], [f"{group}/sin_u"], group=group, axes=[-1])
    s4 = g.node("Tile", [s4, g.const(f"{group}/tile2b", np.array([1, 1, 1, 2], np.int64))],
                [f"{group}/sin_tile"], group=group)
    s4 = g.node("Reshape", [s4, g.const(f"{group}/sin_shape", np.array([0, 0, rk], np.int64))],
                [f"{group}/sin_flat"], group=group)
    s4 = g.node("Unsqueeze", [s4], [f"{group}/sin_b"], group=group, axes=[1])
    if sign < 0:
        s4 = g.node("Neg", [s4], [f"{group}/sin_neg"], group=group)

    nope = g.node("Slice", [x, g.const(f"{group}/nope_s", np.array([0], np.int64)),
                            g.const(f"{group}/nope_e", np.array([dh - rk], np.int64)),
                            g.const(f"{group}/ax3", np.array([3], np.int64))],
                  [f"{group}/nope"], group=group)
    rope = g.node("Slice", [x, g.const(f"{group}/rope_s", np.array([dh - rk], np.int64)),
                            g.const(f"{group}/rope_e", np.array([dh], np.int64)),
                            g.const(f"{group}/ax3b", np.array([3], np.int64))],
                  [f"{group}/rope"], group=group)
    x1 = g.node("Slice", [rope, g.const(f"{group}/x1_s", np.array([0], np.int64)),
                          g.const(f"{group}/x1_e", np.array([rk], np.int64)),
                          g.const(f"{group}/ax3c", np.array([3], np.int64)),
                          g.const(f"{group}/step2", np.array([2], np.int64))],
                [f"{group}/x1"], group=group)
    x2 = g.node("Slice", [rope, g.const(f"{group}/x2_s", np.array([1], np.int64)),
                          g.const(f"{group}/x2_e", np.array([rk], np.int64)),
                          g.const(f"{group}/ax3d", np.array([3], np.int64)),
                          g.const(f"{group}/step2b", np.array([2], np.int64))],
                [f"{group}/x2"], group=group)
    neg = g.node("Neg", [x2], [f"{group}/neg_x2"], group=group)
    p1 = g.node("Unsqueeze", [neg], [f"{group}/neg_u"], group=group, axes=[-1])
    p2 = g.node("Unsqueeze", [x1], [f"{group}/x1_u"], group=group, axes=[-1])
    stack = g.node("Concat", [p1, p2], [f"{group}/stack"], group=group, axis=-1)
    rot = g.node("Reshape", [stack, g.const(f"{group}/rot_shape", np.array([0, 0, 0, rk], np.int64))],
                 [f"{group}/rot"], group=group)
    a = g.node("Mul", [rope, c4], [f"{group}/rope_cos"], group=group)
    b = g.node("Mul", [rot, s4], [f"{group}/rot_sin"], group=group)
    rotated = g.node("Add", [a, b], [f"{group}/rotated"], group=group)
    return g.node("Concat", [nope, rotated], [out], group=group, axis=3)


def clip(g: GB, group: str, x: str, out: str, lo: float, hi: float) -> str:
    """torch.clamp → Clip（opset 18 的 Clip(x, min, max)）。"""
    return g.node("Clip", [x, g.const(f"{out}/clip_lo", np.float32(lo)),
                           g.const(f"{out}/clip_hi", np.float32(hi))], [out], group=group)


def silu(g: GB, group: str, x: str, out: str) -> str:
    """SiLU = x * sigmoid(x)（opset 18 无 SiLU 基础算子）。"""
    s = g.node("Sigmoid", [x], [f"{group}/sig"], group=group)
    return g.node("Mul", [x, s], [out], group=group)
