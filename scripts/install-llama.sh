#!/usr/bin/env bash
# Install the llama.cpp build Nyx AI Core runs as its worker: PrismML's fork
# (see docs/PRISM.md). Replaces the build only when the pinned tag changes, and
# takes Chat's copy of the same release instead of downloading it again.
set -euo pipefail

tag=prism-b10826-c9c34bc
asset=llama-prism-b10826-c9c34bc-bin-linux-cuda-13.3-x64.tar.gz
digest=43e203be88b8c2fbebb24d518078c3ff02df742970d89f739e9dc23646c6ff2d
libdir="${NYX_AI_LIB:-$HOME/.local/lib/nyx-ai/llama.cpp}"
chatdir="$HOME/.local/lib/nyx-chat/llama.cpp"

current() { [[ -x $1/llama-server && -f $1/.release && $(cat "$1/.release") == "$tag" ]]; }

if current "$libdir"; then
  exit 0
fi
install -d "$libdir"
if current "$chatdir"; then
  echo "==> Taking llama.cpp $tag from Nyx Chat's copy"
  cp -a --reflink=auto "$chatdir/." "$libdir/"
  rm -f "$libdir/llama-server-launch"
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
cp -a "$work/llama-$tag/"*.so* "$libdir/"
install -m 755 "$work/llama-$tag/llama-server" "$libdir/llama-server"
printf '%s\n' "$tag" >"$libdir/.release"
