#!/bin/sh
set -eu

launch_agent="$HOME/Library/LaunchAgents/com.pang.ai-investor.monitor.plist"
launchctl bootout "gui/$(id -u)" "$launch_agent" >/dev/null 2>&1 || true
if [ -f "$launch_agent" ]; then
  rm "$launch_agent"
fi
echo "Removed com.pang.ai-investor.monitor. Existing local logs were preserved."
