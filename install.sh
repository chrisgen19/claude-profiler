#!/usr/bin/env bash
# install.sh — Installer for claude-profiler
# Links token_profiler.py into ~/.claude*/hooks/, ~/.local/bin/claude-profiler,
# registers async lifecycle hooks in all ~/.claude*/settings.json files, and runs initial sync.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILER_SCRIPT="$REPO_DIR/token_profiler.py"

chmod +x "$PROFILER_SCRIPT"

mkdir -p "$HOME/.local/bin"
ln -sf "$PROFILER_SCRIPT" "$HOME/.local/bin/claude-profiler"
echo "✓ Linked CLI -> ~/.local/bin/claude-profiler"

for d in "$HOME"/.claude "$HOME"/.claude-*; do
  [ -d "$d" ] || continue
  mkdir -p "$d/hooks"
  ln -sf "$PROFILER_SCRIPT" "$d/hooks/token_profiler.py"
  echo "✓ Linked hook -> $d/hooks/token_profiler.py"
done

python3 - <<'PY'
import json
from pathlib import Path

home = Path.home()
hook_cmd = "python3 ~/.claude/hooks/token_profiler.py hook"
events_to_hook = ["PostToolUse", "SubagentStop", "Stop", "SessionEnd"]

for d in sorted(home.glob(".claude*")):
    if not d.is_dir():
        continue
    s_path = d / "settings.json"
    if not s_path.exists():
        continue
    try:
        data = json.loads(s_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    hooks = data.setdefault("hooks", {})
    changed = False
    for ev in events_to_hook:
        ev_list = hooks.setdefault(ev, [])
        already = any(
            any("token_profiler.py" in (h.get("command") or "") for h in item.get("hooks", []))
            for item in ev_list
            if isinstance(item, dict)
        )
        if not already:
            ev_list.append({
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{hook_cmd} {ev}",
                        "async": True
                    }
                ]
            })
            changed = True
    if changed:
        s_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"✓ Registered async hooks in {s_path}")
    else:
        print(f"• Hooks already registered in {s_path}")
PY

echo "▸ Running initial transcript sync..."
"$PROFILER_SCRIPT" sync
echo "✓ Done! Run 'claude-profiler' from any terminal."
