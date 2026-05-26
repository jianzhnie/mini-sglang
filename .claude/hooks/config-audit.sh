#!/bin/bash
# ConfigChange hook: audit configuration modifications
set -euo pipefail

INPUT=$(cat)
AUDIT_LOG="${CLAUDE_PROJECT_DIR}/.claude/config-audit.log"

jq -c '{
    timestamp: now | todate,
    source: .source,
    file: .file_path,
    change_type: .change_type
}' <<<"$INPUT" >> "$AUDIT_LOG"

exit 0
