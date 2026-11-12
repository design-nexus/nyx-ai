# Recovery and rollback

## Where things are

| What | Where | Safe to delete? |
|---|---|---|
| Settings | `~/.config/nyx-ai/settings.toml` | Yes; defaults come back (Chat's old choices are imported once) |
| Load-time settings per model | `~/.config/nyx-ai/models.ini` | Yes; models load with llama.cpp's defaults |
| Model list and router presets | `~/.local/state/nyx-ai/registry.json`, `router.ini` | Yes, with the daemon stopped; rebuilt on start |
| Downloaded models | `~/.cache/huggingface/hub` (shared with other tools) | Through `nyx-ai rm` only |
| Folder models | `model_dirs` (`~/.local/share/nyx-ai/models`, the Assistant's and Glimpse's folders) | Never touched by AI Core |
| llama.cpp | `~/.local/lib/nyx-ai/llama.cpp` (`.release` names the tag); NixOS: the Nix store | Reinstall with `scripts/install-llama.sh` |
| Socket, worker socket and key | `$XDG_RUNTIME_DIR/nyx-ai/` | Recreated on start |
| Units | `~/.config/systemd/user/nyx-aid.{socket,service}`; NixOS: the system's user units | |

Logs: `journalctl --user -u nyx-aid`. State: `nyx-ai status` (worker state, last
error, restarts, loaded models) and `nyx-ai status --json`.

## What recovers by itself

- **The worker crashes.** Requests in flight fail with `worker_failed`; the
  supervisor restarts llama.cpp after 1, 2, 4, 8 s. Five crashes within 120 s mark
  it failed: requests get `worker_failed` with llama.cpp's last error for a
  minute, then the next request tries again. Models reload on their next use.
- **The daemon stops or crashes.** systemd's socket keeps listening; the next
  request starts `nyx-aid` again. Its worker dies with it (process group and
  parent-death signal), so no orphaned llama.cpp keeps memory.
- **A download stops** (network, reboot, `nyx-aid` restart). The partial file
  stays as `.incomplete`; `nyx-ai pull` again resumes it, and automatic updates
  resume on their own. `nyx-ai cancel-pull MODEL` drops the partial files.
- **Memory runs short.** Idle models unload under memory pressure; a load that
  can't fit fails with `resource_exhausted` instead of pushing the system into
  swap.
- **An app disconnects.** Its requests are cancelled and its action offers end.
  A model it pinned ("keep loaded") stays until the app unloads it or AI Core
  restarts.

## When it doesn't come back

1. `nyx-ai status`. "llama.cpp isn't installed": run
   `~/Projects/nyx-ai/scripts/install-llama.sh` (or `./install.sh`). "keeps
   crashing": the last error usually names the model or the driver.
2. A model that won't load: `nyx-ai settings MODEL` shows its load-time settings;
   remove a bad one with `nyx-ai settings MODEL KEY=` (an empty value), or delete
   its section from `models.ini`.
3. Restart cleanly: `systemctl --user restart nyx-aid.socket` and
   `systemctl --user stop nyx-aid.service`; the next request starts it fresh.
4. Reset state: stop both units, delete `~/.local/state/nyx-ai`, start the
   socket. Models, folders and settings stay.
5. The socket refuses connections ("isn't a private socket"): another program
   made `$XDG_RUNTIME_DIR/nyx-ai`; stop `nyx-aid` and delete that folder.

## Rolling back

### AI Core on the laptop

AI Core installs from the checkout:

```sh
cd ~/Projects/nyx-ai && git checkout <revision> && ./install.sh
```

The installer restarts the socket, so the next request runs the older build.
`git checkout main` and `./install.sh` again returns to the newest.

Compatibility: protocol version 1 only grows. An older app (Chat built against
an older `nyx-ai-client`) works with a newer daemon. A newer app against an older
daemon gets `invalid_request` for fields or operations the old daemon doesn't know
(`grammar`, `list_actions`, image `embed` inputs), so roll back apps first when
rolling the daemon back past a feature they use.

### llama.cpp

`scripts/install-llama.sh` pins one Prism release (`tag`, `asset`, `digest`).
To return to an earlier one, put the earlier three values back (see the history
of `scripts/install-llama.sh` and `docs/PRISM.md`) and run the script; it
replaces the build only when the tag changes. Models that need the newer build
(EmbeddingGemma 2 needs b12128 or later) stop loading on an older one.

### Model updates

A model update downloads the new revision next to the old one and switches
`refs/main` only when the model is idle; blobs nothing links to are pruned after
the switch, so the previous revision isn't kept. To stop updates, set
`nyx-ai service updates=notify` (report only) or `updates=off`. An update that
breaks a model: `nyx-ai rm MODEL` and pull it again, or keep a known-good copy
in a folder in `model_dirs`, which AI Core never updates or deletes.

### NyxOS

- The whole system: boot the previous generation from the boot menu, or
  `sudo nixos-rebuild switch --rollback`.
- AI Core alone: `nyx.ai.enable = false` removes the socket and service; Chat
  then says Nyx AI Core isn't running.
- The CUDA build misbehaving on an NVIDIA system: `nyx.ai.acceleration = "vulkan"`
  uses the Vulkan build instead (no unfree packages).
- A newer `nyx-packages` pin: revert the pin in NyxOS's `flake.lock` and rebuild.

### Chat's old runner

Chat no longer runs its own llama.cpp. Its installer removed
`nyx-llama-server.service` and `~/.local/lib/nyx-chat/llama.cpp`. Rolling Chat
back to a version from before the migration brings its own runner back with its
own installer; it and AI Core would then both use port 8080, so turn one off
(`nyx-ai service openai=off` or Chat's server).
