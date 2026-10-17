#!/usr/bin/env python3
"""structure_reader 薄入口：版本检查后转发到 structure_reader.cli.main。

仅标准库；不导入模型代码，不安装依赖。工程从本脚本相对位置定位
（``../src`` 加入 sys.path），与调用方 cwd 无关。

用法::

    python3 scripts/read_model_structure.py inspect --model deepseek_v4_flash_0731
    python3 scripts/read_model_structure.py --help
"""

import sys

MIN_PYTHON = (3, 11)


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for chunk in text.split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def main() -> int:
    if sys.version_info[:2] < MIN_PYTHON:
        print(
            f"错误: structure_reader 需要 Python {'.'.join(map(str, MIN_PYTHON))} 或更高版本，"
            f"当前为 {sys.version.split()[0]}（{sys.executable}）。"
            "请改用新版解释器运行本脚本。",
            file=sys.stderr,
        )
        return 1
    script_dir = __import__("pathlib").Path(__file__).resolve().parent
    src_dir = script_dir.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        from structure_reader.cli import main as cli_main
    except ImportError as exc:
        print(f"错误: 无法导入 structure_reader 包（已尝试 {src_dir}）: {exc}", file=sys.stderr)
        return 1
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
