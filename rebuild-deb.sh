#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cat "chronos-studio-standalone_0.1.0_amd64.deb.part1" "chronos-studio-standalone_0.1.0_amd64.deb.part2" > "chronos-studio-standalone_0.1.0_amd64.deb"
echo "Rebuilt installer:"
sha256sum "chronos-studio-standalone_0.1.0_amd64.deb"
echo "Expected hash (from .sha256 sidecar):"
cat "chronos-studio-standalone_0.1.0_amd64.deb.sha256"
if sha256sum -c "chronos-studio-standalone_0.1.0_amd64.deb.sha256"; then
  echo "OK - matches the original checksum. Install with:"
  echo "  sudo apt install --reinstall ./chronos-studio-standalone_0.1.0_amd64.deb"
else
  echo "WARNING - checksum mismatch." >&2
fi
