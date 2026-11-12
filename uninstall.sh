#!/usr/bin/env bash
# Remove Nyx AI Core. Models (the Hugging Face cache and model folders) and
# ~/.config/nyx-ai stay.
set -euo pipefail

units="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
systemctl --user disable --now nyx-aid-openai.socket nyx-aid.socket nyx-aid.service 2>/dev/null || true
rm -f "$units/nyx-aid.socket" "$units/nyx-aid-openai.socket" "$units/nyx-aid.service"
systemctl --user daemon-reload 2>/dev/null || true
rm -f "$HOME/.local/bin/nyx-aid" "$HOME/.local/bin/nyx-ai"
rm -rf "$HOME/.local/lib/nyx-ai"
echo "Removed Nyx AI Core. Models and settings in ~/.config/nyx-ai are kept."
