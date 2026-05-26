#!/bin/bash
# Reload environment variables when directory or .envrc/.env changes
set -euo pipefail

if command -v direnv &>/dev/null; then
    direnv export bash 2>/dev/null | grep -v '^export .*\(SECRET\|TOKEN\|PASSWORD\|KEY\)' > "$CLAUDE_ENV_FILE" || true
elif [[ -f .env ]]; then
    grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env | grep -vi '\(SECRET\|TOKEN\|PASSWORD\|KEY\)' > "$CLAUDE_ENV_FILE" 2>/dev/null || true
fi

exit 0
