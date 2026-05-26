#!/bin/bash
#!/bin/bash
# PostToolUse hook: 对编辑过的 .sh 文件自动运行 shellcheck
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(jq -r '.tool_input.file_path // empty' <<<"$INPUT")

# 仅检查 .sh 文件
[[ "$FILE_PATH" != *.sh ]] && exit 0

# 文件不存在则跳过
[[ ! -f "$FILE_PATH" ]] && exit 0

RESULT=$(shellcheck "$FILE_PATH" 2>&1)
EXIT_CODE=$?

if [[ "$EXIT_CODE" -ne 0 ]]; then
    jq -nc --arg msg "shellcheck 发现问题（$FILE_PATH）:\n$RESULT" \
        '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $msg}}'
fi

exit 0
