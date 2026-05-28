#!/bin/bash
# SessionStart compact hook: re-inject critical context after compaction
set -euo pipefail

PROJECT_NAME=$(basename "$CLAUDE_PROJECT_DIR")

echo "## Post-compaction context restore"
echo ""
echo "### Project: ${PROJECT_NAME}"
if [[ -f "${CLAUDE_PROJECT_DIR}/CLAUDE.md" ]]; then
    head -1 "${CLAUDE_PROJECT_DIR}/CLAUDE.md" | sed 's/^# *//'
fi
echo ""

echo "### Key conventions"
echo "- Git: Conventional Commits (feat/fix/docs/refactor/test/chore)"
echo "- Python: ruff, black(88), isort, mypy; 3.10+ syntax, collections.abc over typing"
echo "- Shell: set -euo pipefail, [[ ]] not [ ], double-quote vars, bash 4.2+"
echo "- Hooks: pre-tool-use (security) + post-tool-use (format+shellcheck) on every edit"
echo "- Agents: 9 custom sub-agents available (architect, debugger, code-reviewer, etc.)"
echo ""

echo "### State"
echo "- Branch: $(git -C "$CLAUDE_PROJECT_DIR" branch --show-current 2>/dev/null || echo 'N/A')"
UNCOMMITTED=$(git -C "$CLAUDE_PROJECT_DIR" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
echo "- Uncommitted: ${UNCOMMITTED} files"
echo ""

echo "### Recent commits"
git -C "$CLAUDE_PROJECT_DIR" log --oneline -5 2>/dev/null || echo "(no git history)"

exit 0
