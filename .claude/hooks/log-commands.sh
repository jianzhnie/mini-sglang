#!/bin/bash
# PostToolUse hook: log every Bash command to a file
set -euo pipefail

INPUT=$(cat)
COMMAND=$(jq -r '.tool_input.command // empty' <<<"$INPUT")

[[ -z "$COMMAND" ]] && exit 0

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $COMMAND" >> "${CLAUDE_PROJECT_DIR}/.claude/command-log.txt"
exit 0
