#!/bin/bash
# Notification hook: desktop alert when Claude needs attention
set -euo pipefail

osascript -e 'display notification "Claude Code needs your attention" with title "Claude Code"' 2>/dev/null || true
exit 0
