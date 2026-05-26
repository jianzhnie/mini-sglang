#!/bin/bash
# PreToolUse hook: 阻止危险操作
set -euo pipefail

INPUT=$(cat)
TOOL_NAME=$(jq -r '.tool_name' <<<"$INPUT")

case "$TOOL_NAME" in
    Bash)
        COMMAND=$(jq -r '.tool_input.command' <<<"$INPUT")
        # 阻止 rm -rf / /home /etc 等系统关键目录
        if echo "$COMMAND" | grep -qE 'rm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+/($|\s|home|etc|usr|var|opt)'; then
            jq -n '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: "阻止 rm -rf 删除系统关键目录：该操作被安全策略禁止"}}'
            exit 0
        fi
        # 阻止 rm -rf ~ / $HOME
        # shellcheck disable=SC2016  # 匹配字面量 $HOME 字符串
        if echo "$COMMAND" | grep -qE 'rm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+(\$HOME|~)'; then
            jq -n '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: "阻止删除 HOME 目录：该操作被安全策略禁止"}}'
            exit 0
        fi
        # 阻止写入 .env 文件
        if echo "$COMMAND" | grep -qE '>\s*(\.\s*)?\.env'; then
            jq -n '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: "阻止写入 .env 文件：可能泄露敏感信息，请使用其他方式管理密钥"}}'
            exit 0
        fi
        # 阻止 git push --force / git push -f
        if echo "$COMMAND" | grep -qE 'git\s+push\s+.*(-f\b|--force\b|--force-with-lease\b)'; then
            jq -n '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: "阻止 force push：强制推送会覆盖远程历史，请手动执行"}}'
            exit 0
        fi
        # 阻止推送到 main/master
        if echo "$COMMAND" | grep -qE 'git\s+push\s+.*\b(main|master)\b'; then
            jq -n '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: "阻止推送到 main/master：请通过 PR 合并，避免直接推送主分支"}}'
            exit 0
        fi
        ;;
    Write|Edit)
        FILE_PATH=$(jq -r '.tool_input.file_path' <<<"$INPUT")
        # 阻止写入 .env 文件
        if [[ "$FILE_PATH" == */.env || "$FILE_PATH" == */.env.* ]]; then
            jq -n --arg fp "$FILE_PATH" '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: ("阻止写入 " + $fp + "：.env 文件可能包含敏感信息")}}'
            exit 0
        fi
        ;;
esac

exit 0
