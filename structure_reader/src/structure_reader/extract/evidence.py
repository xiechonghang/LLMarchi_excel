"""来源证据机制（工作包 B）：固定基线身份核对 + 逐条证据的原文核对。

两级机制：
1. :class:`EvidenceBase`：读取项目基线，对 SOURCE_FILES 各输入真实复算
   SHA256 并与固定基线逐条核对——失配具名拒绝（输入变了图作废）。
2. :class:`EvidenceRegistry`：每条证据一条记录（qualified symbol、真实行区间、
   嵌入原文片段、核对结论、等级），构建时对模型源码逐条复核对原文（exact
   substring 匹配，忽略行尾空白）——原文不在行区间内即具名拒绝，不允许
   "有行号就声称语义核对"。

证据等级（coverage）：
- located：符号定位 + 行区间 + 原文核对通过——该证据锚定的行确实含该构造。
- source_checked：该行/区块已逐句核对语义（比 located 强；用于关键控制流，
  如 hc_post 的 comb 方向、HCA 固定压缩位置选择）。
- derived：由已 located/source_checked 的构造经声明式推导得到（如 D_nope、
  D2），推导式记录在 derivation 字段。
- uncovered：诚实标注未覆盖（如 kernel.py 的 sparse_attn 内部语义）。

本模块只读模型源码文本做核对，不导入/不执行模型代码。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from structure_reader.rules.deepseek_v4_flash_0731 import SOURCE_FILES

__all__ = [
    "EvidenceError",
    "SourceEvidence",
    "EvidenceBase",
    "EvidenceRegistry",
    "load_evidence_base",
]

# SOURCE_FILES 键 → 基线 role（与 configs/source_baselines 的 role 一致）。
_BASELINE_ROLES = {
    "model.py": "reference_model_source",
    "kernel.py": "reference_kernel_source",
    "run_config.json": "run_config",
    "root_config.json": "root_config",
    "generate.py": "reference_entrypoint_source",
}

_BASELINE_RELPATH = (
    "structure_reader/configs/source_baselines/"
    "deepseek_v4_flash_0731_reference.json"
)

COVERAGE_LEVELS = ("located", "source_checked", "derived", "uncovered")


class EvidenceError(ValueError):
    """证据核对失败：基线失配、行区间越界或原文不匹配（具名拒绝）。"""


@dataclass(frozen=True)
class SourceEvidence:
    """一条来源证据：qualified symbol、真实行区间、原文片段与核对结论。

    coverage 限 located | source_checked | derived | uncovered 四级：
    - located/source_checked：必须给行区间与原文片段，原文在区间内逐行核对。
    - derived：必须给 derivation（由已核对构造推导的声明式说明）。
    - uncovered：必须给 note，说明未覆盖内容与原因；不允许行号/原文。
    """

    evidence_id: str
    source_key: str  # SOURCE_FILES 键（model.py / kernel.py / run_config.json / ...）
    qualified_symbol: str
    line_start: int | None = None
    line_end: int | None = None
    snippet: str | None = None
    claim: str = ""
    coverage: str = "located"
    derivation: str | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source_key": self.source_key,
            "qualified_symbol": self.qualified_symbol,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "snippet": self.snippet,
            "claim": self.claim,
            "coverage": self.coverage,
            "derivation": self.derivation,
            "note": self.note,
        }


def _load_baseline(workspace_root: Path) -> dict:
    baseline_path = workspace_root / _BASELINE_RELPATH
    if not baseline_path.is_file():
        raise EvidenceError(f"固定基线文件不存在: {baseline_path}")
    return json.loads(baseline_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class EvidenceBase:
    """输入身份基线：各源文件的工作区路径与固定 SHA256（已真实复算核对）。"""

    file_sha256: dict[str, str]  # SOURCE_FILES 键 → 工作区文件实际 SHA256

    def sha256_of(self, source_key: str) -> str:
        try:
            return self.file_sha256[source_key]
        except KeyError as exc:
            raise EvidenceError(f"未知源文件键: {source_key!r}") from exc


def load_evidence_base(workspace_root: Path) -> EvidenceBase:
    """复算 SOURCE_FILES 各输入 SHA256 并与固定基线逐条核对；失配具名拒绝。

    基线内声明的每个角色对应的路径必须真实存在且哈希一致；本包引用的
    SOURCE_FILES 全部角色都必须在基线内声明。任何一条失配即抛
    EvidenceError——输入变化后旧证据不可信，图不得继续构建。
    """
    workspace_root = Path(workspace_root)
    baseline = _load_baseline(workspace_root)
    by_role = {entry["role"]: entry for entry in baseline["source_files"]}
    sha_by_key: dict[str, str] = {}
    for source_key, relpath in SOURCE_FILES.items():
        role = _BASELINE_ROLES[source_key]
        entry = by_role.get(role)
        if entry is None:
            raise EvidenceError(
                f"固定基线缺少 role {role!r}（源文件键 {source_key}）的声明"
            )
        path = workspace_root / relpath
        if not path.is_file():
            raise EvidenceError(f"源文件不存在: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        declared = entry["sha256"]
        if actual != declared:
            raise EvidenceError(
                f"输入基线失配: {relpath} 实际 SHA256 {actual} ≠ 固定基线 {declared}；"
                "输入已变化，既有证据作废，不得继续构建"
            )
        sha_by_key[source_key] = actual
    return EvidenceBase(file_sha256=sha_by_key)


class EvidenceRegistry:
    """证据注册表：注册时逐条对照源文件原文核对，通过后提供只读查询。"""

    def __init__(self, evidence_base: EvidenceBase, workspace_root: Path) -> None:
        self._base = evidence_base
        self._root = Path(workspace_root)
        self._lines_cache: dict[str, list[str]] = {}
        self._evidence: dict[str, SourceEvidence] = {}

    def register(self, evidence: SourceEvidence) -> None:
        """注册一条证据并立即核对；失败具名拒绝，注册表不留半条。"""
        if evidence.evidence_id in self._evidence:
            raise EvidenceError(f"证据 ID 重复: {evidence.evidence_id!r}")
        if evidence.source_key not in SOURCE_FILES:
            raise EvidenceError(f"未知源文件键: {evidence.source_key!r}")
        if evidence.coverage not in COVERAGE_LEVELS:
            raise EvidenceError(
                f"证据 {evidence.evidence_id} 的 coverage 必须是 "
                f"{'/'.join(COVERAGE_LEVELS)}，得到 {evidence.coverage!r}"
            )
        if evidence.coverage in ("located", "source_checked"):
            self._verify_snippet(evidence)
        elif evidence.coverage == "derived":
            if not (evidence.derivation and evidence.derivation.strip()):
                raise EvidenceError(
                    f"证据 {evidence.evidence_id} 为 derived 等级，必须给出 derivation"
                )
            if evidence.snippet is not None or evidence.line_start is not None:
                raise EvidenceError(
                    f"证据 {evidence.evidence_id} 为 derived 等级，不允许行号/原文片段"
                    "（如引用已核对构造，请单独给 located 证据）"
                )
        else:  # uncovered
            if not (evidence.note and evidence.note.strip()):
                raise EvidenceError(
                    f"证据 {evidence.evidence_id} 为 uncovered 等级，必须给出 note"
                )
            if evidence.snippet is not None or evidence.line_start is not None:
                raise EvidenceError(
                    f"证据 {evidence.evidence_id} 为 uncovered 等级，不允许行号/原文片段"
                )
        self._evidence[evidence.evidence_id] = evidence

    def _verify_snippet(self, evidence: SourceEvidence) -> None:
        """行区间越界与原文核对：snippet 每一行（rstrip 后）必须作为子串出现
        在 [line_start, line_end] 区间内的某一行上（子串匹配以容纳 JSON 长行；
        仍锚定在声明区间内，不得越界声称）。"""
        if (
            evidence.line_start is None
            or evidence.line_end is None
            or evidence.snippet is None
        ):
            raise EvidenceError(
                f"证据 {evidence.evidence_id} 为 {evidence.coverage} 等级，"
                "必须同时给出行区间与原文片段"
            )
        if evidence.line_start < 1 or evidence.line_end < evidence.line_start:
            raise EvidenceError(
                f"证据 {evidence.evidence_id} 行区间非法: "
                f"{evidence.line_start}-{evidence.line_end}"
            )
        lines = self._source_lines(evidence.source_key)
        if evidence.line_end > len(lines):
            raise EvidenceError(
                f"证据 {evidence.evidence_id} 行区间越界: 文件共 {len(lines)} 行，"
                f"区间上界 {evidence.line_end}"
            )
        snippet_lines = [
            raw.strip() for raw in evidence.snippet.splitlines() if raw.strip()
        ]
        if not snippet_lines:
            raise EvidenceError(f"证据 {evidence.evidence_id} 的原文片段为空")
        window = [
            line.strip()
            for line in lines[evidence.line_start - 1 : evidence.line_end]
        ]
        for expected in snippet_lines:
            if not any(expected in actual for actual in window):
                raise EvidenceError(
                    f"证据 {evidence.evidence_id} 原文核对失败: 片段行 "
                    f"{expected!r} 未出现在 {SOURCE_FILES[evidence.source_key]}"
                    f":{evidence.line_start}-{evidence.line_end} 内；"
                    "不得凭行号声称语义核对"
                )

    def _source_lines(self, source_key: str) -> list[str]:
        if source_key not in self._lines_cache:
            path = self._root / SOURCE_FILES[source_key]
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise EvidenceError(f"源文件不可读: {path}: {exc}") from exc
            self._lines_cache[source_key] = text.splitlines()
        return self._lines_cache[source_key]

    def get(self, evidence_id: str) -> SourceEvidence:
        try:
            return self._evidence[evidence_id]
        except KeyError as exc:
            raise EvidenceError(f"证据未注册: {evidence_id!r}") from exc

    def __contains__(self, evidence_id: str) -> bool:
        return evidence_id in self._evidence

    def all(self) -> tuple[SourceEvidence, ...]:
        return tuple(self._evidence[eid] for eid in sorted(self._evidence))

    def by_coverage(self, coverage: str) -> tuple[SourceEvidence, ...]:
        return tuple(e for e in self.all() if e.coverage == coverage)
