#!/bin/bash
# SessionEnd hook: send session summary to WeChat via Server酱
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
SESSION_FILE="${PROJECT_DIR}/.claude/.session-start"
COMMAND_LOG="${PROJECT_DIR}/.claude/command-log.txt"

# Load SCT_SENDKEY
SCT_SENDKEY=""
if [[ -f "$PROJECT_DIR/.claude/.env" ]]; then
    SCT_SENDKEY=$(grep '^SCT_SENDKEY=' "$PROJECT_DIR/.claude/.env" | cut -d= -f2-)
fi
[[ -z "${SCT_SENDKEY:-}" ]] && exit 0

# --- Collect summary data ---

# Duration
if [[ -f "$SESSION_FILE" ]]; then
    START_TIME=$(cat "$SESSION_FILE")
    END_TIME=$(date +%s)
    DURATION_SEC=$((END_TIME - START_TIME))
    if (( DURATION_SEC < 60 )); then
        DURATION="${DURATION_SEC}秒"
    else
        DURATION="$((DURATION_SEC / 60))分$((DURATION_SEC % 60))秒"
    fi
    rm -f "$SESSION_FILE"
else
    DURATION="未知"
fi

# Commands run this session (delta from session start)
CMD_COUNT=0
CMD_START_FILE="${PROJECT_DIR}/.claude/.session-cmd-start"
if [[ -f "$COMMAND_LOG" && -f "$CMD_START_FILE" ]]; then
    START_LINES=$(cat "$CMD_START_FILE")
    END_LINES=$(wc -l < "$COMMAND_LOG" | tr -d ' ')
    CMD_COUNT=$((END_LINES - START_LINES))
    [[ $CMD_COUNT -lt 0 ]] && CMD_COUNT=0
    rm -f "$CMD_START_FILE"
fi

# Git changes summary
cd "$PROJECT_DIR"
UNCOMMITTED=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
DIFF_STAT=$(git diff --stat --color=never 2>/dev/null | tail -1 || echo "")
CHANGED_FILES=$( (git diff --name-only 2>/dev/null; git diff --cached --name-only 2>/dev/null) | sort -u | head -20)

# Edited files this session
EDIT_LOG="${PROJECT_DIR}/.claude/.session-edits"
EDIT_SUMMARY=""
if [[ -f "$EDIT_LOG" && -s "$EDIT_LOG" ]]; then
    TOTAL_EDITS=$(wc -l < "$EDIT_LOG" | tr -d ' ')
    # Group by file: count occurrences, list unique files
    EDIT_FILES=$(awk -F'|' '{
        files[$1]++
    } END {
        for (f in files) printf "- `%s` (%d次)\n", f, files[f]
    }' "$EDIT_LOG" | sort)
    EDIT_SUMMARY="**编辑操作**: ${TOTAL_EDITS} 次，涉及 $(echo "$EDIT_FILES" | wc -l | tr -d ' ') 个文件
${EDIT_FILES}"
    rm -f "$EDIT_LOG"
fi
COMMITS=""
if [[ -n "${START_TIME:-}" ]]; then
    SINCE_TS=$(date -r "$START_TIME" -u +%Y-%m-%dT%H:%M:%S 2>/dev/null || echo "")
    if [[ -n "$SINCE_TS" ]]; then
        COMMITS=$(git log --oneline --since="$SINCE_TS" 2>/dev/null | head -5 || echo "")
    fi
fi

# --- Build WeChat message ---

BRANCH=$(git branch --show-current 2>/dev/null || echo "N/A")
PROJECT=$(basename "$PROJECT_DIR")

TITLE="Claude Code · ${PROJECT} · ${BRANCH}"

DESP="**会话时长**: ${DURATION}
**分支**: ${BRANCH}
**执行命令**: ${CMD_COUNT} 条"

if [[ -n "$EDIT_SUMMARY" ]]; then
    DESP="${DESP}
${EDIT_SUMMARY}"
fi

DESP="${DESP}
**未提交变更**: ${UNCOMMITTED} 个文件"

if [[ -n "$DIFF_STAT" ]]; then
    DESP="${DESP}
**变更统计**: ${DIFF_STAT}"
fi

if [[ -n "$COMMITS" ]]; then
    DESP="${DESP}
**本次提交**:
${COMMITS}"
fi

if [[ "$UNCOMMITTED" -gt 0 ]]; then
    DESP="${DESP}
**变更文件**:
$(echo "$CHANGED_FILES" | awk '{print "- " $0}')"
fi

# Truncate if too long (WeChat template message limit)
MAX_LEN=2000
if [[ ${#DESP} -gt $MAX_LEN ]]; then
    DESP="${DESP:0:$MAX_LEN}..."
fi

# Send via Server酱
curl -s -X POST "https://sctapi.ftqq.com/${SCT_SENDKEY}.send" \
    -d "title=${TITLE}" \
    -d "desp=${DESP}" \
    -o /dev/null

# Cleanup temp files
rm -f "${PROJECT_DIR}"/.claude/tmp-*.txt 2>/dev/null || true

exit 0
