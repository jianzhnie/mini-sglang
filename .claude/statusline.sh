#!/bin/bash
# Status line: context usage + git branch/changes + model
# Must NOT use set -e — any failure should degrade gracefully, not blank the line.
set -uo pipefail

input=$(cat)

MODEL=$(echo "$input" | jq -r '.model.display_name // "?"' 2>/dev/null || echo "?")

# Safely extract context percentage as integer, defaulting to 0
raw=$(echo "$input" | jq -r '.context_window.used_percentage // 0' 2>/dev/null || echo "0")
PCT=$(echo "$raw" | cut -d. -f1)
# Validate PCT is a non-negative integer, default to 0 otherwise
if ! [[ "$PCT" =~ ^[0-9]+$ ]]; then
    PCT=0
fi

# Context bar (10 chars)
BAR_WIDTH=10
FILLED=$((PCT * BAR_WIDTH / 100))
[[ $FILLED -gt $BAR_WIDTH ]] && FILLED=$BAR_WIDTH
[[ $FILLED -lt 0 ]] && FILLED=0
printf -v bar_fill "%${FILLED}s"
printf -v bar_pad "%$((BAR_WIDTH - FILLED))s"
BAR="${bar_fill// /▓}${bar_pad// /░}"

# Color: green <50%, yellow <80%, red >=80%
if (( PCT >= 80 )); then
    COLOR='\033[31m'
elif (( PCT >= 50 )); then
    COLOR='\033[33m'
else
    COLOR='\033[32m'
fi
RESET='\033[0m'

# Context display
printf "%b%s %s %3d%%%b" "$COLOR" "$BAR" "$MODEL" "$PCT" "$RESET"

# Git info
if git rev-parse --git-dir >/dev/null 2>&1; then
    BRANCH=$(git branch --show-current 2>/dev/null || echo "?")
    STAGED=$(git diff --cached --name-only 2>/dev/null | wc -l | tr -d ' ')
    MODIFIED=$(git diff --name-only 2>/dev/null | wc -l | tr -d ' ')
    UNTRACKED=$(git ls-files --others --exclude-standard 2>/dev/null | wc -l | tr -d ' ')

    printf "  \033[36m%s\033[0m" "$BRANCH"
    [[ "${STAGED:-0}" -gt 0 ]] && printf " \033[32m+%d\033[0m" "$STAGED"
    [[ "${MODIFIED:-0}" -gt 0 ]] && printf " \033[33m~%d\033[0m" "$MODIFIED"
    [[ "${UNTRACKED:-0}" -gt 0 ]] && printf " \033[90m?%d\033[0m" "$UNTRACKED"
fi
