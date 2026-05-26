#!/bin/bash
# SessionStart compact hook: re-inject critical context after compaction
set -euo pipefail

PROJECT_NAME=$(basename "$CLAUDE_PROJECT_DIR")

echo "## Post-compaction context restore"
echo ""
echo "### Project: ${PROJECT_NAME}"
if [[ -f "${CLAUDE_PROJECT_DIR}/CLAUDE.md" ]]; then
    head -3 "${CLAUDE_PROJECT_DIR}/CLAUDE.md" | grep -v '^$' | head -1 | sed 's/^#* *//'
fi
echo ""
echo "### Key conventions"
echo "- Git: Conventional Commits (feat/fix/docs/refactor/test/chore)"
echo "- Shell: set -euo pipefail, double-quote vars, [[ ]] not [ ]"
echo "- Compat: bash 4.2+, no declare -A, no namerefs"
echo "- Auto-format: pre-commit runs on every Edit/Write (isort, ruff format, ruff lint)"
echo "- Protected: package-lock.json, .git/ (proctect-files); .env, force push, main push, rm -rf (block-dangerous)"
echo ""
echo "### Recent commits"
git -C "$CLAUDE_PROJECT_DIR" log --oneline -5 2>/dev/null || echo "(no git history)"
echo ""
echo "### Current branch"
git -C "$CLAUDE_PROJECT_DIR" branch --show-current 2>/dev/null || echo "(unknown)"
