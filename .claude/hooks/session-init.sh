#!/bin/bash
# SessionStart hook: initialize session with project context
set -euo pipefail

PROJECT_NAME=$(basename "$CLAUDE_PROJECT_DIR")

# Record session start time for summary
date -u +%s > "${CLAUDE_PROJECT_DIR}/.claude/.session-start"

# Record initial command log line count for session summary
COMMAND_LOG="${CLAUDE_PROJECT_DIR}/.claude/command-log.txt"
if [[ -f "$COMMAND_LOG" ]]; then
    wc -l < "$COMMAND_LOG" | tr -d ' ' > "${CLAUDE_PROJECT_DIR}/.claude/.session-cmd-start"
else
    echo "0" > "${CLAUDE_PROJECT_DIR}/.claude/.session-cmd-start"
fi

# Clear session edit log
:> "${CLAUDE_PROJECT_DIR}/.claude/.session-edits"

echo "## Session started: ${PROJECT_NAME}"
echo ""
echo "Branch: $(git -C "$CLAUDE_PROJECT_DIR" branch --show-current 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version 2>/dev/null || echo 'N/A')"
echo ""
# Show uncommitted changes so Claude is aware of repo state
changes=$(git -C "$CLAUDE_PROJECT_DIR" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
if [[ "$changes" -gt 0 ]]; then
    echo "Uncommitted changes: $changes file(s)"
    git -C "$CLAUDE_PROJECT_DIR" status --short 2>/dev/null || true
fi

exit 0
