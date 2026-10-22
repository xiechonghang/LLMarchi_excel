#!/bin/bash
# 项目内最小构建/注册脚本。默认只构建到 structure_reader/.work/tools/netron_launcher08。
# 注意：本阶段禁止注册 scheme；register/unregister 仅供后续已授权阶段使用。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "$HERE/../../.." && pwd)"          # 项目根
OUT="$PROJ/structure_reader/.work/tools/netron_launcher08"
APP="$OUT/LLMarchiNetron.app"
LSREG="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"

build() {
  mkdir -p "$APP/Contents/MacOS" "$OUT/tmp" "$OUT/.cache"
  TMPDIR="$OUT/tmp" /usr/bin/swiftc -O -module-cache-path "$OUT/.cache" \
    "$HERE/main.swift" -o "$APP/Contents/MacOS/LLMarchiNetron" \
    -framework AppKit -framework CryptoKit
  sed "s#__PROJECT_ROOT__#$PROJ#" "$HERE/Info.plist" > "$APP/Contents/Info.plist"
  echo "built: $APP"
}

case "${1:-build}" in
  build) build ;;
  register)
    if [ "${2:-}" != "--yes" ]; then
      echo "拒绝执行：注册 scheme 需显式 --yes，且应在本阶段授权后运行" >&2; exit 2
    fi
    build
    "$LSREG" -f "$APP"
    echo "registered: llmarchi-netron" ;;
  unregister)
    "$LSREG" -u "$APP" && echo "unregistered" ;;
  *)
    echo "usage: $0 [build|register --yes|unregister]" >&2; exit 2 ;;
esac
