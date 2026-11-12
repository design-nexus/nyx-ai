#!/usr/bin/env bash
# Install Nyx AI Core: nyx-aid (the shared local model runtime), the nyx-ai
# command, the llama.cpp worker, and the user units that start it on first use.
#
#   ./install.sh          build this checkout and install it
#
# NYX_AI_NO_ENABLE=1 installs without enabling the socket.
set -euo pipefail

REPO="design-nexus/nyx-ai"
say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m==>\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31m==>\033[0m %s\n' "$*" >&2; exit 1; }

case ${1:-ai} in
  ai | all | --*) ;;
  -h | --help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) die "This repository installs only ai (Nyx AI Core)." ;;
esac

here=""
[[ -n ${BASH_SOURCE[0]:-} && -f ${BASH_SOURCE[0]} ]] && here=$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# ---------- Source ----------
prebuilt=false
if [[ -n $here && -f $here/.prebuilt ]]; then
  src=$here
  prebuilt=true
elif [[ -n $here && -f $here/Cargo.toml && -d $here/crates ]]; then
  src=$here
else
  say "Fetching the source"
  git clone -q --depth 1 "https://github.com/$REPO.git" "$work/src"
  src="$work/src"
fi

# ---------- Build ----------
target=${CARGO_TARGET_DIR:-$src/target}
if [[ $prebuilt == false ]]; then
  if ! command -v cargo >/dev/null; then
    command -v pacman >/dev/null || die "Rust (cargo) is needed to build Nyx AI Core."
    say "Installing rust"
    sudo pacman -S --needed --noconfirm rust
  fi
  say "Building Nyx AI Core"
  (cd "$src" && cargo build --release --locked -p nyx-aid -p nyx-ai)
fi

# ---------- Install ----------
bin="$HOME/.local/bin"
units="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$bin" "$units"
say "Installing nyx-aid and nyx-ai"
for b in nyx-aid nyx-ai; do
  if [[ -f $target/release/$b ]]; then
    install -m 755 "$target/release/$b" "$bin/$b"
  else
    install -m 755 "$src/$b" "$bin/$b"
  fi
done
install -m 644 "$src/data/nyx-aid.service" "$src/data/nyx-aid.socket" "$src/data/nyx-aid-openai.socket" "$units/"
bash "$src/scripts/install-llama.sh"
systemctl --user daemon-reload 2>/dev/null || true

if [[ ${NYX_AI_NO_ENABLE:-0} != 1 ]]; then
  systemctl --user enable --now nyx-aid.socket >/dev/null 2>&1 || warn "Couldn't enable nyx-aid.socket; run: systemctl --user enable --now nyx-aid.socket"
fi
# A running daemon would keep the old build and the socket its old addresses;
# both start again on the next request.
if systemctl --user is-active -q nyx-aid.socket 2>/dev/null; then
  systemctl --user stop nyx-aid.service 2>/dev/null || true
  systemctl --user restart nyx-aid.socket || warn "Couldn't restart nyx-aid.socket"
fi

case ":$PATH:" in
  *":$bin:"*) ;;
  *) warn "$bin isn't on your PATH; nyx-ai won't be found by name." ;;
esac
say "Done. Nyx AI Core starts when an app first uses it; try: nyx-ai status"
