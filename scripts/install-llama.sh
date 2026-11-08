#!/usr/bin/env bash
# Install the llama.cpp build Nyx AI Core runs as its worker: PrismML's fork
# (see docs/PRISM.md). Replaces the build only when the pinned tag changes;
# scripts/check-llama.sh lists newer releases.
set -euo pipefail

tag=prism-b12128-d7b2896
asset=llama-prism-b12128-d7b2896-bin-linux-cuda-13.3-x64.tar.gz
digest=9f3a8d2de2b69936b8ec7c994579ad419cb2918785483d80091b6b2776b880bd
libdir="${NYX_AI_LIB:-$HOME/.local/lib/nyx-ai/llama.cpp}"

current() { [[ -x $1/llama-server && -f $1/.release && $(cat "$1/.release") == "$tag" ]]; }

if current "$libdir"; then
  exit 0
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
if [[ -n ${NYX_LLAMA_ARCHIVE:-} ]]; then
  cp "$NYX_LLAMA_ARCHIVE" "$work/$asset"
else
  echo "==> Downloading llama.cpp $tag (PrismML)"
  curl -fL --retry 3 -o "$work/$asset" "https://github.com/PrismML-Eng/llama.cpp/releases/download/$tag/$asset"
fi
printf '%s  %s\n' "$digest" "$work/$asset" | sha256sum -c -
tar -xzf "$work/$asset" -C "$work"
# Build into a fresh folder and swap it in, so a running worker keeps its files.
fresh="$libdir.new"
rm -rf "$fresh"
install -d "$fresh"
cp -a "$work/llama-$tag/"*.so* "$fresh/"
install -m 755 "$work/llama-$tag/llama-server" "$fresh/llama-server"
printf '%s\n' "$tag" >"$fresh/.release"
rm -rf "$libdir.old"
[[ -d $libdir ]] && mv "$libdir" "$libdir.old"
mv "$fresh" "$libdir"
rm -rf "$libdir.old"
