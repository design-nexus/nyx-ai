#!/usr/bin/env bash
# List PrismML llama.cpp releases newer than the one install-llama.sh pins, with
# each Linux asset's size and the SHA-256 GitHub publishes. Changes nothing; to
# move the pin, copy the tag, asset and digest into install-llama.sh and run the
# checks in docs/PRISM.md.
set -euo pipefail

here=$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
pinned=$(sed -n 's/^tag=//p' "$here/install-llama.sh")
echo "Pinned: $pinned"
curl -fsSL "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases?per_page=30" | python3 -c '
import json, sys
pinned = sys.argv[1]
newer = []
for r in json.load(sys.stdin):
    if r["draft"] or r["prerelease"]:
        continue
    if r["tag_name"] == pinned:
        break
    newer.append(r)
if not newer:
    print("No newer release.")
for r in newer:
    tag, day = r["tag_name"], r["published_at"][:10]
    print(f"\n{tag}  ({day})")
    for a in r["assets"]:
        name, size = a["name"], a["size"] // 1_000_000
        if "linux" in name or "ubuntu" in name:
            digest = (a.get("digest") or "").removeprefix("sha256:")
            print(f"  {name}  {size} MB  {digest}")
' "$pinned"
