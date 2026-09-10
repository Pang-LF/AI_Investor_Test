#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
executable="$project_root/.venv/bin/ai-investor"
template="$project_root/config/com.pang.ai-investor.monitor.plist.template"
launch_agent_dir="$HOME/Library/LaunchAgents"
target="$launch_agent_dir/com.pang.ai-investor.monitor.plist"
runtime_dir="$project_root/.local/launchd"

if [ ! -x "$executable" ]; then
  echo "Missing $executable; create the virtual environment and install the project first." >&2
  exit 1
fi

mkdir -p "$launch_agent_dir" "$runtime_dir"

PROJECT_ROOT="$project_root" \
EXECUTABLE="$executable" \
TEMPLATE="$template" \
TARGET="$target" \
RUNTIME_DIR="$runtime_dir" \
"$project_root/.venv/bin/python" - <<'PY'
import os
from pathlib import Path

text = Path(os.environ["TEMPLATE"]).read_text(encoding="utf-8")
replacements = {
    "__AI_INVESTOR_EXECUTABLE__": os.environ["EXECUTABLE"],
    "__AI_INVESTOR_ROOT__": os.environ["PROJECT_ROOT"],
    "__AI_INVESTOR_STDOUT__": str(Path(os.environ["RUNTIME_DIR"]) / "stdout.log"),
    "__AI_INVESTOR_STDERR__": str(Path(os.environ["RUNTIME_DIR"]) / "stderr.log"),
}
for placeholder, value in replacements.items():
    text = text.replace(placeholder, value)
Path(os.environ["TARGET"]).write_text(text, encoding="utf-8")
PY

launchctl bootout "gui/$(id -u)" "$target" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$target"
launchctl enable "gui/$(id -u)/com.pang.ai-investor.monitor"

echo "Installed com.pang.ai-investor.monitor (every 900 seconds)."
echo "The Python market-hours gate prevents Robinhood calls outside regular hours."
