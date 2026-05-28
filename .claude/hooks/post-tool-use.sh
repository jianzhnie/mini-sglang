#!/bin/bash
# PostToolUse hook: auto-format edited files and shellcheck .sh files
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(jq -r '.tool_input.file_path // empty' <<<"$INPUT")
TOOL_NAME=$(jq -r '.tool_name // "Edit"' <<<"$INPUT")

# File must exist
[[ ! -f "$FILE_PATH" ]] && exit 0

# Log edited file for session summary (deduplicate per session)
EDIT_LOG="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/.claude/.session-edits"
echo "${FILE_PATH}|${TOOL_NAME}" >> "$EDIT_LOG"

OUTPUT=""

# Step 1: run pre-commit auto-format on all edited files
FORMAT_RESULT=$(pre-commit run --files "$FILE_PATH" 2>&1) || true
if [[ -n "$FORMAT_RESULT" ]]; then
    OUTPUT+="${FORMAT_RESULT}"
fi

# Step 2: run shellcheck on .sh files
if [[ "$FILE_PATH" == *.sh ]]; then
    SHELLCHECK_RESULT=$(shellcheck "$FILE_PATH" 2>&1) || true
    if [[ -n "$SHELLCHECK_RESULT" ]]; then
        [[ -n "$OUTPUT" ]] && OUTPUT+=$'\n'
        OUTPUT+="${SHELLCHECK_RESULT}"
    fi
fi

# Report aggregated results
if [[ -n "$OUTPUT" ]]; then
    jq -nc --arg msg "$OUTPUT" \
        '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $msg}}'
fi

exit 0
