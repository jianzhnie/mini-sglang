#!/bin/bash
# PostToolUse hook: auto-format edited files with pre-commit
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(jq -r '.tool_input.file_path // empty' <<<"$INPUT")

# Skip if file doesn't exist
[[ ! -f "$FILE_PATH" ]] && exit 0

RESULT=$(pre-commit run --files "$FILE_PATH" 2>&1) || true

jq -nc --arg msg "$(printf '%s' "$RESULT")" \
    '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $msg}}'

exit 0
