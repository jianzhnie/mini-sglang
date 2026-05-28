#!/bin/bash
# SubagentStart hook: log subagent spawn events
set -euo pipefail

INPUT=$(cat)
AGENT_TYPE=$(jq -r '.agent_type // "unknown"' <<<"$INPUT")

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] agent:spawn ${AGENT_TYPE}" >> "${PROJECT_DIR}/.claude/command-log.txt"
exit 0
