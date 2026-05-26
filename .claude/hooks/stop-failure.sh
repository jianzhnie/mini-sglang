#!/bin/bash
# StopFailure hook: log API errors during session
set -euo pipefail

INPUT=$(cat)
ERROR_TYPE=$(jq -r '.error_type // "unknown"' <<<"$INPUT")

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] StopFailure: $ERROR_TYPE" >> "${CLAUDE_PROJECT_DIR}/.claude/stop-failure.log"
exit 0
