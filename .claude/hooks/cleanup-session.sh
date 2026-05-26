#!/bin/bash
# SessionEnd hook: clean up temp files on /clear
set -euo pipefail

rm -f "${CLAUDE_PROJECT_DIR}"/.claude/tmp-*.txt 2>/dev/null || true
exit 0
