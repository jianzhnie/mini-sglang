#!/bin/bash
# PostToolUse hook: log every Bash command to a file
set -euo pipefail

INPUT=$(cat)
COMMAND=$(jq -r '.tool_input.command // empty' <<<"$INPUT")

[[ -z "$COMMAND" ]] && exit 0

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $COMMAND" >> "${PROJECT_DIR}/.claude/command-log.txt"
exit 0
