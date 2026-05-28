#!/bin/bash
# PreToolUse hook: block dangerous commands and protect sensitive files
set -euo pipefail

INPUT=$(cat)
TOOL_NAME=$(jq -r '.tool_name' <<<"$INPUT")

# ---- Bash command checks ----
if [[ "$TOOL_NAME" == "Bash" ]]; then
    COMMAND=$(jq -r '.tool_input.command' <<<"$INPUT")

    # Block rm -rf on system-critical directories
    if echo "$COMMAND" | grep -qE 'rm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+/($|\s|home|etc|usr|var|opt)'; then
        jq -n '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: "Blocked: rm -rf on system directory is prohibited by safety policy"}}'
        exit 0
    fi

    # Block rm -rf ~ / $HOME
    if echo "$COMMAND" | grep -qE 'rm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+(\$HOME|~)'; then
        jq -n '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: "Blocked: rm -rf on HOME directory is prohibited by safety policy"}}'
        exit 0
    fi

    # Block writing to .env files via shell redirection
    if echo "$COMMAND" | grep -qE '>\s*(\.\s*)?\.env'; then
        jq -n '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: "Blocked: writing to .env may leak secrets — use a different approach for credentials"}}'
        exit 0
    fi

    # Block force push
    if echo "$COMMAND" | grep -qE 'git\s+push\s+.*(-f\b|--force\b|--force-with-lease\b)'; then
        jq -n '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: "Blocked: force push overwrites remote history — please run manually"}}'
        exit 0
    fi

    # Block direct push to main/master
    if echo "$COMMAND" | grep -qE 'git\s+push\s+.*\b(main|master)\b'; then
        jq -n '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: "Blocked: pushing to main/master — use a PR instead"}}'
        exit 0
    fi
fi

# ---- Write/Edit file path checks ----
if [[ "$TOOL_NAME" == "Write" || "$TOOL_NAME" == "Edit" ]]; then
    FILE_PATH=$(jq -r '.tool_input.file_path // empty' <<<"$INPUT")

    # Block .env files
    if [[ "$FILE_PATH" == */.env || "$FILE_PATH" == */.env.* ]]; then
        jq -n --arg fp "$FILE_PATH" \
            '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: ("Blocked: " + $fp + " — .env files may contain sensitive data")}}'
        exit 0
    fi

    # Block protected file patterns
    readonly PROTECTED_PATTERNS=("package-lock.json" ".git/")
    for pattern in "${PROTECTED_PATTERNS[@]}"; do
        if [[ "$FILE_PATH" == *"$pattern"* ]]; then
            jq -n --arg fp "$FILE_PATH" --arg pat "$pattern" \
                '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: ("Blocked: " + $fp + " matches protected pattern " + $pat)}}'
            exit 0
        fi
    done
fi

exit 0
